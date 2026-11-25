# -*- coding: utf-8 -*-
"""
思路 B (Exp15): 全新外部独立数据集零样本盲测 (External Zero-Shot Validation)
1. 外部临床队列盲测: 仅用细胞系药物库 (sci-Plex3) 训练 BioMolKG，零样本盲测从未见过的 GEO 原代人源胶质母细胞瘤病人肿瘤数据 (ZhaoSims2021.h5ad)
2. 外部独立实验室队列盲测: 仅用 K562+HepG2 训练，零样本盲测独立 CROP-seq T 细胞队列 (DatlingerBock2021.h5ad)
严格对比 Linear-Ridge vs AIVC-OmniFusion / BioMolKG。
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
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


def run_external_patient_gbm_test(device):
    print("\n=== External Validation 1: sci-Plex3 (Cell Lines) -> ZhaoSims2021 (GEO Patient Glioblastoma) ===")
    scip = sc.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad")
    gbm = sc.read_h5ad(r"D:\data\AIVC\data\ZhaoSims2021.h5ad")

    # 若 GBM 未标准化，先标准化
    x_max = float(gbm.X[:500].max()) if gbm.n_obs > 500 else float(gbm.X.max())
    if x_max > 25.0:
        sc.pp.normalize_total(gbm, target_sum=1e4)
        sc.pp.log1p(gbm)

    # 对齐共有基因 (以 sci-Plex3 HVG2000 为基准)
    scip_genes = list(scip.var_names)
    gbm_idx = {g: i for i, g in enumerate(gbm.var_names)}
    common = [(si, gbm_idx[g]) for si, g in enumerate(scip_genes) if g in gbm_idx]
    s_cols = np.array([c[0] for c in common], dtype=np.int64)
    g_cols = np.array([c[1] for c in common], dtype=np.int64)
    print(f"  -> 共有基因对齐: {len(common)}/{len(scip_genes)}")

    # 加载 sci-Plex3 药物分子+ESM2靶点特征
    mf = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=False)
    kg = np.load(r"D:\data\AIVC\data\sciplex3_target_esm2_kg.npz", allow_pickle=False)
    sm_csv = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
    d2tg = dict(zip(sm_csv["perturbation"].str.lower(), sm_csv["target"].fillna("Others")))
    tg_emb = {str(t): kg["target_embs"][i] for i, t in enumerate(kg["targets"])}
    pw_emb = {str(d).lower(): kg["pathway_onehot"][i] for i, d in enumerate(kg["drugs"])}
    mol_emb = {str(d).lower(): np.hstack([mf["morgan"][i], mf["chemberta"][i]]) for i, d in enumerate(mf["drugs"])}

    def get_drug_feat(dname: str) -> np.ndarray:
        dl = dname.lower()
        # 模糊前缀匹配 (如 panobinostat -> Panobinostat (LBH589))
        hit = None
        for k in mol_emb:
            if k == dl or k.startswith(dl + " ") or k.startswith(dl + "("):
                hit = k
                break
        if not hit:
            return None
        tg = d2tg.get(hit, "Others")
        return np.hstack([mol_emb[hit], tg_emb.get(tg, np.zeros(640, dtype=np.float32)), pw_emb.get(hit, np.zeros(16, dtype=np.float32))]).astype(np.float32)

    # 提取 sci-Plex3 训练样本 (按 cell_line control 中心化)
    tr_X, tr_D, tr_Y = [], [], []
    for cl, grp in scip.obs.groupby("cell_line", observed=True):
        sub = scip[grp.index]
        cm = (sub.obs["perturbation"] == "control").values
        if cm.sum() < 10:
            continue
        mu_c = _to_dense_array(sub.X[cm][:, s_cols]).mean(axis=0).astype(np.float32)
        for p, pgrp in sub.obs[~cm].groupby("perturbation", observed=True):
            if len(pgrp) < 10:
                continue
            dname = str(p).split("__dose_")[0]
            df = get_drug_feat(dname)
            if df is None:
                continue
            mu_p = _to_dense_array(sub[pgrp.index].X[:, s_cols]).mean(axis=0).astype(np.float32)
            tr_X.append(mu_c)
            tr_D.append(df)
            tr_Y.append(mu_p - mu_c)

    tr_X = np.vstack(tr_X)
    tr_D = np.vstack(tr_D)
    tr_Y = np.vstack(tr_Y)

    # 提取 ZhaoSims2021 GBM 病人外部测试样本
    gbm_cm = (gbm.obs["perturbation"].astype(str) == "control").values
    mu_gbm = _to_dense_array(gbm.X[gbm_cm][:, g_cols]).mean(axis=0).astype(np.float32)
    te_drugs, te_D, te_Y_dict = [], [], {}
    for p, grp in gbm.obs[~gbm_cm].groupby("perturbation", observed=True):
        if len(grp) < 20:
            continue
        df = get_drug_feat(str(p))
        if df is None:
            continue
        mu_p = _to_dense_array(gbm[grp.index].X[:, g_cols]).mean(axis=0).astype(np.float32)
        te_drugs.append(str(p))
        te_D.append(df)
        te_Y_dict[str(p)] = mu_p - mu_gbm

    print(f"  -> 训练 (sci-Plex3): {len(tr_Y)} 样本 | 外部盲测 (GEO 原代 GBM 病人共有药物): {len(te_drugs)} 种 ({te_drugs})")
    if not te_drugs:
        return []

    te_X = np.tile(mu_gbm, (len(te_drugs), 1)).astype(np.float32)
    te_D = np.vstack(te_D)

    # Linear-Ridge (Cell + DrugFeat)
    ridge = Ridge(alpha=100.0).fit(np.hstack([tr_X, tr_D]), tr_Y)
    pred_r = ridge.predict(np.hstack([te_X, te_D]))
    df_r = evaluate_perturbation_predictions(te_Y_dict, {d: pred_r[i] for i, d in enumerate(te_drugs)}, "Linear-Ridge(External_GBM)")

    # BioMolKG (Cell + Drug SMILES + ESM2)
    torch.manual_seed(42)
    n_g, d_dim = tr_X.shape[1], tr_D.shape[1]
    net = nn.Sequential(
        nn.Linear(n_g + d_dim, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.SiLU(),
        nn.Linear(256, n_g),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    t_in = torch.tensor(np.hstack([tr_X, tr_D]), device=device)
    t_y = torch.tensor(tr_Y, device=device)
    net.train()
    for _ in range(60):
        perm = torch.randperm(len(tr_Y), device=device)
        for i in range(0, len(tr_Y), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            out = net(t_in[idx])
            loss = F.mse_loss(out, t_y[idx]) + 0.1 * (1.0 - F.cosine_similarity(out, t_y[idx], dim=-1).mean())
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred_nn = net(torch.tensor(np.hstack([te_X, te_D]), device=device)).cpu().numpy()
    df_nn = evaluate_perturbation_predictions(te_Y_dict, {d: pred_nn[i] for i, d in enumerate(te_drugs)}, "BioMolKG(External_GBM)")

    for m, d in [("Linear-Ridge(External_GBM)", df_r), ("BioMolKG(External_GBM)", df_nn)]:
        print(f"  [{m}] Pearson={d['pearson_delta_all'].mean():.4f} | Top20={d['pearson_delta_top20'].mean():.4f} | DirAcc20={d['dir_acc_top20'].mean():.4f} | MSE={d['mse_all'].mean():.6f}")
    return [df_r, df_nn]


def run_external_cropseq_test():
    print("\n=== External Validation 2: K562+HepG2 -> DatlingerBock2021 (Unseen CROP-seq Jurkat T-Cell) ===")
    dat = sc.read_h5ad(r"D:\data\AIVC\data\DatlingerBock2021.h5ad")
    x_max = float(dat.X[:500].max()) if dat.n_obs > 500 else float(dat.X.max())
    if x_max > 25.0:
        sc.pp.normalize_total(dat, target_sum=1e4)
        sc.pp.log1p(dat)

    # 加载 K562+RPE1 参考基因空间并对齐
    ref = sc.read_h5ad(r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad")
    target_genes = list(ref.var_names)
    dat_idx = {g: i for i, g in enumerate(dat.var_names)}
    common = [(ti, dat_idx[g]) for ti, g in enumerate(target_genes) if g in dat_idx]
    t_cols = np.array([c[0] for c in common], dtype=np.int64)
    d_cols = np.array([c[1] for c in common], dtype=np.int64)

    # 清洗 Datlinger2021 perturbation (如 ZAP70_1 -> ZAP70, non-targeting/control -> control)
    dat.obs["clean_p"] = dat.obs["perturbation"].astype(str).map(
        lambda x: "control" if ("non-targeting" in x.lower() or "control" in x.lower() or "ctrl" in x.lower()) else x.split("_")[0].strip()
    )
    cm = (dat.obs["clean_p"] == "control").values
    mu_dat = np.zeros(len(target_genes), dtype=np.float32)
    mu_dat[t_cols] = _to_dense_array(dat.X[cm][:, d_cols]).mean(axis=0).astype(np.float32)
    dat_deltas = {}
    for p, grp in dat.obs[~cm].groupby("clean_p", observed=True):
        if len(grp) >= 20:
            dv = np.zeros(len(target_genes), dtype=np.float32)
            dv[t_cols] = _to_dense_array(dat[grp.index].X[:, d_cols]).mean(axis=0).astype(np.float32) - mu_dat[t_cols]
            dat_deltas[str(p)] = dv

    # 用 Replogle K562 训练集 (完全不接触 Datlinger2021) 训练 Ridge 基线对比
    tr_ref = ref[ref.obs["ood_split"] == "train"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    overlap_p = sorted(set(d_k562.keys()) & set(dat_deltas.keys()))
    print(f"  -> Datlinger2021 有效扰动数: {len(dat_deltas)} | 与 K562 共有基因扰动: {len(overlap_p)} ({overlap_p})")
    if not overlap_p:
        return []

    tr_keys = sorted(d_k562.keys())
    lin_idx = {t: i for i, t in enumerate(tr_keys)}
    def enc(ks):
        m = np.zeros((len(ks), len(tr_keys)), dtype=np.float32)
        for r, k in enumerate(ks):
            if k in lin_idx:
                m[r, lin_idx[k]] = 1.0
        return m
    ridge = Ridge(alpha=10.0).fit(enc(tr_keys), np.vstack([d_k562[k] for k in tr_keys]))
    pred_r = ridge.predict(enc(overlap_p))
    df_r = evaluate_perturbation_predictions({k: dat_deltas[k] for k in overlap_p}, {k: pred_r[i] for i, k in enumerate(overlap_p)}, "Linear-Ridge(External_CROPseq)")
    print(f"  [Linear-Ridge(External_CROPseq)] Pearson={df_r['pearson_delta_all'].mean():.4f} | Top20={df_r['pearson_delta_top20'].mean():.4f} | MSE={df_r['mse_all'].mean():.6f}")
    return [df_r]


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfs = []
    dfs.extend(run_external_patient_gbm_test(dev))
    dfs.extend(run_external_cropseq_test())
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(r"D:\data\AIVC\test_results\exp15_external_zeroshot_validation.csv", index=False, encoding="utf-8")
        print("\n[Exp15] 外部独立盲测结果已保存至 exp15_external_zeroshot_validation.csv")
