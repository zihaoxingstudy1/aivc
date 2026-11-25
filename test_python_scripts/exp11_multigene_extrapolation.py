# -*- coding: utf-8 -*-
"""
方向 7 (Exp11): 3 个及以上基因多点同时扰动组合外推 (1+2 -> 3+ Zero-Shot Combinatorial Extrapolation)
基于 TianKampmann2019_iPSC.h5ad (275,708 正常人源 iPSC 细胞):
- 对所有多基因组合按基因名排序去重标准化 (防止 A_B_C 与 B_A_C 顺序泄漏)
- 训练集: 仅使用 1-gene 与 2-gene 扰动
- 测试集: 完全未见过的 3+ 基因多点同时扰动 (>=20 细胞)
对比 Linear-Ridge (线性加和) vs AIVC-OmniFusion (Set-Attention 5-Teacher 融合, max_len=6)
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from omni_fusion import GatedMultiTeacherFusionNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def canonical_pert(p: str) -> str:
    if p == "control":
        return "control"
    toks = sorted({t.strip() for t in str(p).split("_") if t.strip() and t.strip() != "control"})
    return "_".join(toks) if toks else "control"


def run_exp11():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_h5ad = r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad"
    hvg_cache = r"D:\data\AIVC\data\Tian2019_iPSC_combo_hvg2000.h5ad"

    if os.path.exists(hvg_cache):
        print(f"[Exp11] 加载缓存 HVG2000: {hvg_cache}")
        adata = sc.read_h5ad(hvg_cache)
    else:
        print(f"[Exp11] 读取并标准化 {raw_h5ad} ...")
        adata = sc.read_h5ad(raw_h5ad)
        adata.obs["canon_pert"] = adata.obs["perturbation"].astype(str).map(canonical_pert)
        adata.obs["n_canon"] = adata.obs["canon_pert"].map(lambda x: 0 if x == "control" else len(x.split("_")))
        # 保留 control 以及有效细胞数 >= 20 的组合 (n_canon in 1..6)
        vc = adata.obs["canon_pert"].value_counts()
        valid_p = set(vc[vc >= 20].index)
        keep = (adata.obs["canon_pert"] == "control") | (adata.obs["canon_pert"].isin(valid_p) & adata.obs["n_canon"].between(1, 6))
        adata = adata[keep].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        adata.write_h5ad(hvg_cache)
        print(f"[Exp11] 缓存完成: {adata.shape}")

    obs = adata.obs.copy()
    X_all = _to_dense_array(adata.X).astype(np.float32)
    ctrl_mask = (obs["canon_pert"] == "control").values
    mu_ctrl = X_all[ctrl_mask].mean(axis=0)

    # 构建排序去重后的各组 Delta
    deltas = {}
    n_genes_map = {}
    for cp, grp in obs[~ctrl_mask].groupby("canon_pert", observed=True):
        idx = obs.index.get_indexer(grp.index)
        deltas[str(cp)] = X_all[idx].mean(axis=0) - mu_ctrl
        n_genes_map[str(cp)] = int(grp["n_canon"].iloc[0])

    tr_keys = sorted([k for k, n in n_genes_map.items() if n in (1, 2)])
    # 只测试所有组分单基因都在训练词汇表里出现过的 3+ 基因组合 (严格组合外推)
    tr_vocab = sorted({t for k in tr_keys for t in k.split("_")})
    tr_vocab_set = set(tr_vocab)
    te_keys = sorted([k for k, n in n_genes_map.items() if n >= 3 and all(t in tr_vocab_set for t in k.split("_"))])

    print(f"[Exp11 Split] Train (1+2 gene): {len(tr_keys)} 组 | Test (3+ gene Zero-Shot): {len(te_keys)} 组 | 共有基因词汇: {len(tr_vocab)}")
    for k_n in sorted({n_genes_map[k] for k in te_keys}):
        cnt = sum(1 for k in te_keys if n_genes_map[k] == k_n)
        print(f"  -> Test {k_n}-gene combos: {cnt}")

    # 1. Linear-Ridge (1+2 -> 3+)
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(keys):
        m = np.zeros((len(keys), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(keys):
            for t in ps.split("_"):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True).fit(enc_lin(tr_keys), np.vstack([deltas[k] for k in tr_keys]))
    pred_ridge = ridge.predict(enc_lin(te_keys))
    te_true_dict = {k: deltas[k] for k in te_keys}
    df_ridge = evaluate_perturbation_predictions(te_true_dict, {k: pred_ridge[i] for i, k in enumerate(te_keys)}, "Linear-Ridge(1+2->3+)")

    # 2. AIVC-OmniFusion (5-Teacher Set-Attention, max_len=6)
    all_vocab = sorted(set(tr_vocab) | {t for k in te_keys for t in k.split("_")})
    t2id = {t: i + 1 for i, t in enumerate(all_vocab)}
    max_len = 6

    var_ens = {g: str(adata.var["ensemble_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensemble_id" in adata.var.columns else {}
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
    coexp_dict = build_gene_coexp_prior(adata[obs["n_canon"] <= 2].X, list(adata.var_names), prior_dim=64)

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}
    teacher_matrices = []
    for td in teachers.values():
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(all_vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

    def make_tids(keys):
        arr = np.zeros((len(keys), max_len), dtype=np.int64)
        for i, p in enumerate(keys):
            for j, t in enumerate(p.split("_")[:max_len]):
                arr[i, j] = t2id.get(t, 0)
        return arr

    tr_tids = torch.tensor(make_tids(tr_keys), device=device)
    tr_dl = torch.tensor(ridge.predict(enc_lin(tr_keys)).astype(np.float32), device=device)
    tr_y = torch.tensor(np.vstack([deltas[k] for k in tr_keys]).astype(np.float32), device=device)
    tr_xc = torch.tensor(np.tile(mu_ctrl, (len(tr_keys), 1)).astype(np.float32), device=device)

    te_tids = torch.tensor(make_tids(te_keys), device=device)
    te_dl = torch.tensor(pred_ridge.astype(np.float32), device=device)
    te_xc = torch.tensor(np.tile(mu_ctrl, (len(te_keys), 1)).astype(np.float32), device=device)

    # 5-Seed Ensemble AIVC-OmniFusion
    preds_ens = []
    for seed in (42, 101, 202, 303, 404):
        torch.manual_seed(seed)
        net = GatedMultiTeacherFusionNet(
            n_genes=adata.n_vars,
            n_tokens=len(all_vocab),
            teacher_matrices=teacher_matrices,
            bottleneck_dim=64,
            hidden_dim=192,
            teacher_ctx_dim=1152,
            dropout=0.20,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        net.train()
        for ep in range(80):
            perm = torch.randperm(len(tr_keys), device=device)
            for i in range(0, len(tr_keys), 64):
                idx = perm[i : i + 64]
                opt.zero_grad()
                pred, _, _ = net(tr_xc[idx], tr_tids[idx], tr_dl[idx])
                loss = F.mse_loss(pred, tr_y[idx]) + 0.1 * (1.0 - F.cosine_similarity(pred, tr_y[idx], dim=-1).mean())
                loss.backward()
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            p_te, _, _ = net(te_xc, te_tids, te_dl)
            preds_ens.append(p_te.cpu().numpy())

    pred_omni = np.mean(preds_ens, axis=0)
    df_omni = evaluate_perturbation_predictions(te_true_dict, {k: pred_omni[i] for i, k in enumerate(te_keys)}, "AIVC-OmniFusion(1+2->3+)")
    res_all = pd.concat([df_ridge, df_omni], ignore_index=True)
    res_all["n_genes_perturbed"] = res_all["perturbation"].map(n_genes_map)
    res_all.to_csv(r"D:\data\AIVC\test_results\exp11_multigene_extrapolation.csv", index=False, encoding="utf-8")

    print("\n=== Exp11: 3+ 基因多点同时扰动 Zero-Shot 外推评测 ===")
    for m in ["Linear-Ridge(1+2->3+)", "AIVC-OmniFusion(1+2->3+)"]:
        sub = res_all[res_all["model"] == m]
        print(f"  [{m}] Pearson={sub['pearson_delta_all'].mean():.4f} | Top20={sub['pearson_delta_top20'].mean():.4f} | DirAcc20={sub['dir_acc_top20'].mean():.4f} | MSE={sub['mse_all'].mean():.6f}")
        for kn, g in sub.groupby("n_genes_perturbed"):
            print(f"    -> {kn}-gene ({len(g)} combos): Pearson={g['pearson_delta_all'].mean():.4f} | Top20={g['pearson_delta_top20'].mean():.4f} | MSE={g['mse_all'].mean():.6f}")


if __name__ == "__main__":
    run_exp11()
