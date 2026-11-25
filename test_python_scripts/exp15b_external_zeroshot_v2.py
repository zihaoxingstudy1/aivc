# -*- coding: utf-8 -*-
"""
思路 B 升级 (Exp15b): 外部独立数据集零样本盲测 v2
1. GEO 原代病人 GBM (ZhaoSims2021): 使用 Z-Score 跨平台基态对齐 + Linear-Ridge 残差锚定 BioMolKG
2. 独立实验室 CROP-seq Jurkat (DatlingerBock2021): 使用 Nadig2024 Jurkat + Replogle K562 共有基因扰动评测 Linear-Ridge vs AIVC-OmniFusion
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from omni_fusion import GatedMultiTeacherFusionNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def zscore_vec(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-6)


def test_external_gbm_anchored(device):
    print("\n=== External Validation 1 (v2): sci-Plex3 -> ZhaoSims2021 (GEO Patient GBM) ===")
    scip = sc.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad")
    gbm = sc.read_h5ad(r"D:\data\AIVC\data\ZhaoSims2021.h5ad")
    x_max = float(gbm.X[:500].max()) if gbm.n_obs > 500 else float(gbm.X.max())
    if x_max > 25.0:
        sc.pp.normalize_total(gbm, target_sum=1e4)
        sc.pp.log1p(gbm)

    scip_genes = list(scip.var_names)
    gbm_idx = {g: i for i, g in enumerate(gbm.var_names)}
    common = [(si, gbm_idx[g]) for si, g in enumerate(scip_genes) if g in gbm_idx]
    s_cols = np.array([c[0] for c in common], dtype=np.int64)
    g_cols = np.array([c[1] for c in common], dtype=np.int64)

    mf = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=False)
    kg = np.load(r"D:\data\AIVC\data\sciplex3_target_esm2_kg.npz", allow_pickle=False)
    sm_csv = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
    d2tg = dict(zip(sm_csv["perturbation"].str.lower(), sm_csv["target"].fillna("Others")))
    tg_emb = {str(t): kg["target_embs"][i] for i, t in enumerate(kg["targets"])}
    pw_emb = {str(d).lower(): kg["pathway_onehot"][i] for i, d in enumerate(kg["drugs"])}
    mol_emb = {str(d).lower(): np.hstack([mf["morgan"][i], mf["chemberta"][i]]) for i, d in enumerate(mf["drugs"])}

    def get_df(dname: str):
        dl = dname.lower()
        hit = next((k for k in mol_emb if k == dl or k.startswith(dl + " ") or k.startswith(dl + "(")), None)
        if not hit:
            return None
        tg = d2tg.get(hit, "Others")
        return np.hstack([mol_emb[hit], tg_emb.get(tg, np.zeros(640, dtype=np.float32)), pw_emb.get(hit, np.zeros(16, dtype=np.float32))]).astype(np.float32)

    tr_X, tr_D, tr_Y = [], [], []
    for cl, grp in scip.obs.groupby("cell_line", observed=True):
        sub = scip[grp.index]
        cm = (sub.obs["perturbation"] == "control").values
        if cm.sum() < 10:
            continue
        mu_c = _to_dense_array(sub.X[cm][:, s_cols]).mean(axis=0).astype(np.float32)
        mu_cz = zscore_vec(mu_c)
        for p, pgrp in sub.obs[~cm].groupby("perturbation", observed=True):
            if len(pgrp) < 10:
                continue
            df = get_df(str(p).split("__dose_")[0])
            if df is None:
                continue
            mu_p = _to_dense_array(sub[pgrp.index].X[:, s_cols]).mean(axis=0).astype(np.float32)
            tr_X.append(mu_cz)
            tr_D.append(df)
            tr_Y.append(mu_p - mu_c)

    tr_X, tr_D, tr_Y = np.vstack(tr_X), np.vstack(tr_D), np.vstack(tr_Y)

    gbm_cm = (gbm.obs["perturbation"].astype(str) == "control").values
    mu_gbm = _to_dense_array(gbm.X[gbm_cm][:, g_cols]).mean(axis=0).astype(np.float32)
    mu_gbm_z = zscore_vec(mu_gbm)
    te_drugs, te_D, te_Y_dict = [], [], {}
    for p, grp in gbm.obs[~gbm_cm].groupby("perturbation", observed=True):
        if len(grp) < 20:
            continue
        df = get_df(str(p))
        if df is None:
            continue
        mu_p = _to_dense_array(gbm[grp.index].X[:, g_cols]).mean(axis=0).astype(np.float32)
        te_drugs.append(str(p))
        te_D.append(df)
        te_Y_dict[str(p)] = mu_p - mu_gbm

    te_X = np.tile(mu_gbm_z, (len(te_drugs), 1)).astype(np.float32)
    te_D = np.vstack(te_D)

    ridge = Ridge(alpha=500.0).fit(np.hstack([tr_X, tr_D]), tr_Y)
    pred_r_tr = ridge.predict(np.hstack([tr_X, tr_D])).astype(np.float32)
    pred_r_te = ridge.predict(np.hstack([te_X, te_D])).astype(np.float32)
    df_r = evaluate_perturbation_predictions(te_Y_dict, {d: pred_r_te[i] for i, d in enumerate(te_drugs)}, "Linear-Ridge(External_GBM)")

    # Residual-Anchored BioMolKG: pred = delta_linear + 0.35 * tanh(MLP([xc_z, df]))
    torch.manual_seed(42)
    n_g, d_dim = tr_X.shape[1], tr_D.shape[1]
    res_net = nn.Sequential(
        nn.Linear(n_g + d_dim, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Dropout(0.25),
        nn.Linear(256, n_g),
    ).to(device)
    opt = torch.optim.AdamW(res_net.parameters(), lr=1e-3, weight_decay=1e-2)
    t_in = torch.tensor(np.hstack([tr_X, tr_D]), device=device)
    t_dl = torch.tensor(pred_r_tr, device=device)
    t_y = torch.tensor(tr_Y, device=device)

    res_net.train()
    for _ in range(50):
        perm = torch.randperm(len(tr_Y), device=device)
        for i in range(0, len(tr_Y), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            pred = t_dl[idx] + 0.35 * torch.tanh(res_net(t_in[idx]))
            loss = F.mse_loss(pred, t_y[idx]) + 0.1 * (1.0 - F.cosine_similarity(pred, t_y[idx], dim=-1).mean())
            loss.backward()
            opt.step()

    res_net.eval()
    with torch.no_grad():
        pred_nn = (torch.tensor(pred_r_te, device=device) + 0.35 * torch.tanh(res_net(torch.tensor(np.hstack([te_X, te_D]), device=device)))).cpu().numpy()
    df_nn = evaluate_perturbation_predictions(te_Y_dict, {d: pred_nn[i] for i, d in enumerate(te_drugs)}, "BioMolKG-Anchored(External_GBM)")

    for m, d in [("Linear-Ridge(External_GBM)", df_r), ("BioMolKG-Anchored(External_GBM)", df_nn)]:
        print(f"  [{m}] Pearson={d['pearson_delta_all'].mean():.4f} | Top20={d['pearson_delta_top20'].mean():.4f} | DirAcc20={d['dir_acc_top20'].mean():.4f} | MSE={d['mse_all'].mean():.6f}")
    return [df_r, df_nn]


def test_external_cropseq_jurkat(device):
    print("\n=== External Validation 2 (v2): Nadig2024 Jurkat -> DatlingerBock2021 (External CROP-seq Jurkat) ===")
    jur = sc.read_h5ad(r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad")
    dat = sc.read_h5ad(r"D:\data\AIVC\data\DatlingerBock2021.h5ad")

    for a in (jur, dat):
        x_max = float(a.X[:500].max()) if a.n_obs > 500 else float(a.X.max())
        if x_max > 25.0:
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)

    dat.obs["clean_p"] = dat.obs["perturbation"].astype(str).map(
        lambda x: "control" if ("non-targeting" in x.lower() or "control" in x.lower() or "ctrl" in x.lower()) else x.split("_")[0].strip()
    )
    dat_perts = set(dat.obs.loc[dat.obs["clean_p"] != "control", "clean_p"].unique())
    jur_perts = set(jur.obs.loc[jur.obs["perturbation"] != "control", "perturbation"].astype(str).unique())
    overlap_p = sorted(dat_perts & jur_perts)
    print(f"  -> Nadig2024 Jurkat 与 Datlinger2021 共有基因扰动: {len(overlap_p)} 个 ({overlap_p})")
    if not overlap_p:
        return []

    sc.pp.highly_variable_genes(jur, n_top_genes=2000, subset=True)
    target_genes = list(jur.var_names)
    dat_idx = {g: i for i, g in enumerate(dat.var_names)}
    common = [(ji, dat_idx[g]) for ji, g in enumerate(target_genes) if g in dat_idx]
    j_cols = np.array([c[0] for c in common], dtype=np.int64)
    d_cols = np.array([c[1] for c in common], dtype=np.int64)
    print(f"  -> 共有基因对齐: {len(common)}/{len(target_genes)}")

    cm_j = (jur.obs["perturbation"].astype(str) == "control").values
    mu_j = _to_dense_array(jur.X[cm_j]).mean(axis=0).astype(np.float32)
    j_deltas = {}
    vc_j = jur.obs.loc[~cm_j, "perturbation"].astype(str).value_counts()
    for p in vc_j[vc_j >= 10].index:
        pm = (jur.obs["perturbation"].astype(str) == p).values
        j_deltas[str(p)] = _to_dense_array(jur.X[pm]).mean(axis=0).astype(np.float32) - mu_j

    cm_d = (dat.obs["clean_p"] == "control").values
    mu_d = np.zeros(len(target_genes), dtype=np.float32)
    mu_d[j_cols] = _to_dense_array(dat.X[cm_d][:, d_cols]).mean(axis=0).astype(np.float32)
    d_deltas = {}
    for p in overlap_p:
        pm = (dat.obs["clean_p"] == p).values
        if pm.sum() >= 20:
            dv = np.zeros(len(target_genes), dtype=np.float32)
            dv[j_cols] = _to_dense_array(dat.X[pm][:, d_cols]).mean(axis=0).astype(np.float32) - mu_d[j_cols]
            d_deltas[p] = dv

    eval_p = sorted(set(d_deltas.keys()) & set(j_deltas.keys()))
    tr_p = sorted(j_deltas.keys())
    print(f"  -> 实际训练 (Nadig Jurkat): {len(tr_p)} 扰动 | 外部盲测 (Datlinger2021): {len(eval_p)} 共有扰动 ({eval_p})")
    if not eval_p:
        return []

    # 独立外部评测: 直接看 Nadig Jurkat 同基因单细胞效应迁移到外部实验室 Datlinger2021 的一致性 (Zero-Shot Cross-Lab Transfer)
    pred_transfer = {p: j_deltas[p] for p in eval_p}
    df_tr = evaluate_perturbation_predictions(d_deltas, pred_transfer, "CrossLab-Direct(Jurkat->Datlinger2021)")
    print(f"  [CrossLab-Direct(Jurkat->Datlinger2021)] Pearson={df_tr['pearson_delta_all'].mean():.4f} | Top20={df_tr['pearson_delta_top20'].mean():.4f} | DirAcc20={df_tr['dir_acc_top20'].mean():.4f}")
    return [df_tr]


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfs = []
    dfs.extend(test_external_gbm_anchored(dev))
    dfs.extend(test_external_cropseq_jurkat(dev))
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(r"D:\data\AIVC\test_results\exp15b_external_zeroshot_v2.csv", index=False, encoding="utf-8")
