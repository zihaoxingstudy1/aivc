# -*- coding: utf-8 -*-
"""
方向 1 终极扩展 (Exp8): 8 谱系超级训练集 (K562 + HepG2 + Jurkat + iPSC + Norman + Primary_T + THP1 -> Unseen RPE1)
以极低内存 (Per-Lineage Pseudobulk Delta 投影到 RPE1 HVG2000 空间) 检验多谱系缩放律对跨细胞系泛化的提升！
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from omni_fusion import GatedMultiTeacherFusionNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def extract_lineage_deltas(
    h5ad_path: str,
    target_genes: list,
    lineage_name: str,
    min_cells: int = 10,
    single_pert_only: bool = False,
):
    print(f"[8-Lineage] 读取 {lineage_name}: {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    if single_pert_only and "nperts" in adata.obs.columns:
        keep = (adata.obs["perturbation"].astype(str) == "control") | (adata.obs["nperts"] == 1)
        adata = adata[keep].copy()

    # 若尚未 log1p 标准化 (max > 25)，则标准化
    x_max = float(adata.X[:500].max()) if adata.n_obs > 500 else float(adata.X.max())
    if x_max > 25.0:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 映射基因到 target_genes (2000 HVG)
    src_idx = {g: i for i, g in enumerate(adata.var_names)}
    common_pairs = [(ti, src_idx[g]) for ti, g in enumerate(target_genes) if g in src_idx]
    t_cols = np.array([p[0] for p in common_pairs], dtype=np.int64)
    s_cols = np.array([p[1] for p in common_pairs], dtype=np.int64)
    print(f"  -> {lineage_name} 细胞数={adata.n_obs}, 共有基因={len(common_pairs)}/{len(target_genes)}")

    obs = adata.obs
    ctrl_mask = (obs["perturbation"].astype(str) == "control").values
    X_sub = _to_dense_array(adata.X[ctrl_mask][:, s_cols]).astype(np.float32)
    mu_ctrl = np.zeros(len(target_genes), dtype=np.float32)
    mu_ctrl[t_cols] = X_sub.mean(axis=0)

    deltas = {}
    vc = obs.loc[~ctrl_mask, "perturbation"].astype(str).value_counts()
    valid_perts = vc[vc >= min_cells].index.tolist()
    X_pert_all = adata.X[:, s_cols]
    for p in valid_perts:
        pmask = (obs["perturbation"].astype(str) == p).values
        Xp = _to_dense_array(X_pert_all[pmask]).astype(np.float32)
        d = np.zeros(len(target_genes), dtype=np.float32)
        d[t_cols] = Xp.mean(axis=0) - mu_ctrl[t_cols]
        deltas[p] = d

    print(f"  -> {lineage_name} 提取有效扰动数: {len(deltas)}")
    return mu_ctrl, deltas


def run_exp8():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    ref = sc.read_h5ad(ref_h5ad)
    target_genes = list(ref.var_names)

    # 1. 从 Replogle_K562_RPE1_hvg2000_split 提取 K562(train) 与 RPE1(test)
    tr_ref = ref[ref.obs["ood_split"] == "train"]
    ev_ref = ref[ref.obs["ood_split"] == "test"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    mu_rpe1 = _to_dense_array(ev_ref[ev_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    d_rpe1 = _compute_group_deltas(ev_ref[ev_ref.obs["perturbation"] != "control"], "perturbation", mu_rpe1)

    coexp_dict = build_gene_coexp_prior(tr_ref.X, target_genes, prior_dim=64)
    var_ens = {g: str(ref.var["ensemble_id"].iloc[i]) for i, g in enumerate(target_genes)} if "ensemble_id" in ref.var.columns else {}
    del ref, tr_ref, ev_ref

    # 2. 提取其余 6 个训练谱系
    lineage_specs = [
        ("K562", mu_k562, d_k562),
    ]
    extra_files = [
        ("HepG2", r"D:\data\AIVC\data\NadigOConner2024_hepg2.h5ad", False),
        ("Jurkat", r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad", False),
        ("iPSC", r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad", True),
        ("Norman", r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad", True),
        ("PrimaryT", r"D:\data\AIVC\data\ShifrutMarson2018.h5ad", True),
        ("THP1", r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad", True),
    ]
    for lname, fpath, sp_only in extra_files:
        if os.path.exists(fpath):
            mc, dd = extract_lineage_deltas(fpath, target_genes, lname, min_cells=10, single_pert_only=sp_only)
            lineage_specs.append((lname, mc, dd))

    # 汇总训练样本 (lineage_name, pert, mu_ctrl, delta)
    tr_samples = []
    for lname, mc, dd in lineage_specs:
        for p, dv in dd.items():
            # 只保留单基因扰动以对齐 RPE1 单基因测试
            if "_" in p or "+" in p:
                continue
            tr_samples.append((lname, p, mc, dv))

    ev_keys = sorted(d_rpe1.keys())
    print(f"\n[8-Lineage Summary] 训练样本总数: {len(tr_samples)} (来自 {len(lineage_specs)} 个独立谱系) -> 测试 RPE1 未见细胞系: {len(ev_keys)} 个扰动")

    # 3. Linear-Ridge (7-Lineage MultiCtx-PerCtrl)
    tr_vocab = sorted({s[1] for s in tr_samples})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    X_tr_lin = enc_lin([s[1] for s in tr_samples])
    Y_tr_lin = np.vstack([s[3] for s in tr_samples])
    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_lin)
    pred_ridge = ridge.predict(enc_lin(ev_keys))
    df_ridge = evaluate_perturbation_predictions(d_rpe1, {k: pred_ridge[i] for i, k in enumerate(ev_keys)}, "Linear-Ridge(7-Lineage)")

    # 4. AIVC-OmniFusion (7-Lineage 5-Teacher)
    all_vocab = sorted(set(tr_vocab) | set(ev_keys))
    t2id = {t: i + 1 for i, t in enumerate(all_vocab)}

    gf_dict = extract_geneformer_gene_embeddings(
        r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl",
        all_vocab,
        var_ens,
    )
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", all_vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", all_vocab)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}
    teacher_matrices = []
    for td in teachers.values():
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(all_vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

    tr_xc = torch.tensor(np.vstack([s[2] for s in tr_samples]).astype(np.float32), device=device)
    tr_tids = torch.tensor([[t2id.get(s[1], 0)] for s in tr_samples], dtype=torch.long, device=device)
    tr_dl = torch.tensor(ridge.predict(X_tr_lin).astype(np.float32), device=device)
    tr_y = torch.tensor(Y_tr_lin.astype(np.float32), device=device)

    ev_xc = torch.tensor(np.tile(mu_rpe1, (len(ev_keys), 1)).astype(np.float32), device=device)
    ev_tids = torch.tensor([[t2id.get(k, 0)] for k in ev_keys], dtype=torch.long, device=device)
    ev_dl = torch.tensor(pred_ridge.astype(np.float32), device=device)

    torch.manual_seed(42)
    net = GatedMultiTeacherFusionNet(
        n_genes=len(target_genes),
        n_tokens=len(all_vocab),
        teacher_matrices=teacher_matrices,
        bottleneck_dim=64,
        hidden_dim=192,
        teacher_ctx_dim=1152,
        dropout=0.20,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    net.train()
    for ep in range(60):
        perm = torch.randperm(len(tr_samples), device=device)
        for i in range(0, len(tr_samples), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            pred, _, _ = net(tr_xc[idx], tr_tids[idx], tr_dl[idx])
            loss = F.mse_loss(pred, tr_y[idx]) + 0.1 * (1.0 - F.cosine_similarity(pred, tr_y[idx], dim=-1).mean())
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    with torch.no_grad():
        pred_omni, _, _ = net(ev_xc, ev_tids, ev_dl)
        pred_omni = pred_omni.cpu().numpy()

    df_omni = evaluate_perturbation_predictions(d_rpe1, {k: pred_omni[i] for i, k in enumerate(ev_keys)}, "AIVC-OmniFusion(7-Lineage)")
    res_all = pd.concat([df_ridge, df_omni], ignore_index=True)
    res_all.to_csv(r"D:\data\AIVC\test_results\exp8_8lineage_rpe1_crosscl.csv", index=False, encoding="utf-8")

    print("\n=== Unseen RPE1 跨细胞系泛化评测 (7 谱系联合训练) ===")
    for m in ["Linear-Ridge(7-Lineage)", "AIVC-OmniFusion(7-Lineage)"]:
        sub = res_all[res_all["model"] == m]
        print(f"  [{m}] Pearson={sub['pearson_delta_all'].mean():.4f} | Top20={sub['pearson_delta_top20'].mean():.4f} | DirAcc20={sub['dir_acc_top20'].mean():.4f} | MSE={sub['mse_all'].mean():.6f}")


if __name__ == "__main__":
    run_exp8()
