# -*- coding: utf-8 -*-
"""
exp34_uq_calibration_on_exp31.py
Exp34: 在 v1.1 底座 (Exp31) 上攻克 Deep Ensemble (5成员) + 生物流形 OOD 距离不确定性校准
检验两大顶刊硬指标：
  1. Uncertainty vs 真实误差 (MSE / 1-Pearson) 的 Spearman 正相关性（攻克此前 r=-0.055 的痛点）
  2. Risk-Coverage 选择性预测曲线：过滤掉高不确定性 Top-20% / Top-50% 样本后，剩余置信样本的精度提升幅度
  3. 5-Member Deep Ensemble 均值是否进一步刷新 Unseen RPE1 跨系盲测纪录
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from bfc import build_gene_coexp_prior
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings
from exp31_lineage_invariant_gnn_proteome import predict_lineage_prot_phos_via_exp28, LineageInvariantPathwayNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp34_uq_calibration_summary.csv"
OUT_DETAIL = r"D:\data\AIVC\test_results\exp34_uq_per_perturbation.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp34_uq_calibration_plot.png"


def main():
    t0 = time.time()
    print(f"=== Exp34: 5-Member Deep Ensemble + OOD Distance UQ Calibration ({DEVICE}) ===")
    ref_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    ref = sc.read_h5ad(ref_h5ad)
    target_genes = list(ref.var_names)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    n_genes = len(target_genes)

    tr_ref = ref[ref.obs["ood_split"] == "train"]
    ev_ref = ref[ref.obs["ood_split"] == "test"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    mu_rpe1 = _to_dense_array(ev_ref[ev_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    d_rpe1 = _compute_group_deltas(ev_ref[ev_ref.obs["perturbation"] != "control"], "perturbation", mu_rpe1)

    coexp_dict = build_gene_coexp_prior(tr_ref.X, target_genes, prior_dim=64)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    del ref, tr_ref, ev_ref

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

    lineage_specs = [("K562", mu_k562, d_k562)]
    for lname, fpath, sp_only in [
        ("HepG2", r"D:\data\AIVC\data\NadigOConner2024_hepg2.h5ad", False),
        ("Jurkat", r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad", False),
        ("iPSC", r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad", True),
        ("Norman", r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad", True),
        ("PrimaryT", r"D:\data\AIVC\data\ShifrutMarson2018.h5ad", True),
        ("THP1", r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad", True),
    ]:
        if os.path.exists(fpath):
            mc, dd = extract_lineage_deltas(fpath, target_genes, lname, min_cells=10, single_pert_only=sp_only)
            lineage_specs.append((lname, mc, dd))

    lineage_ctrls = {lname: mc for lname, mc, _ in lineage_specs}
    lineage_ctrls["RPE1"] = mu_rpe1
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    tr_samples = [(ln, p, mc, dv) for ln, mc, dd in lineage_specs for p, dv in dd.items() if "_" not in p and "+" not in p]
    ev_keys = sorted(d_rpe1.keys())

    tr_vocab = sorted({s[1] for s in tr_samples})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    X_tr_lin = enc_lin([s[1] for s in tr_samples])
    Y_tr_lin = np.vstack([s[3] for s in tr_samples]).astype(np.float32)
    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_lin)
    pred_ridge_ev = ridge.predict(enc_lin(ev_keys)).astype(np.float32)

    pred_ridge_tr = np.zeros_like(Y_tr_lin)
    lineages_tr = np.array([s[0] for s in tr_samples])
    for ln in np.unique(lineages_tr):
        m_out = (lineages_tr == ln)
        if (~m_out).sum() > 0:
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[~m_out], Y_tr_lin[~m_out])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out]).astype(np.float32)

    svd = TruncatedSVD(n_components=64, random_state=42).fit(Y_tr_lin)
    U_prog = svd.components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

    d_coexp = len(next(iter(coexp_dict.values())))
    d_esm = len(next(iter(esm2_dict.values())))

    def build_feats(sample_list, ridge_preds):
        ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
        for idx, (lname, p, mc, _) in enumerate(sample_list):
            ctrl_l.append(mc)
            teach_l.append(np.concatenate([coexp_dict.get(p, np.zeros(d_coexp, np.float32)), esm2_dict.get(p, np.zeros(d_esm, np.float32))]))
            gi = gene_to_idx.get(p, -1)
            gp, gph = prot_phos_map[lname]
            if gi >= 0:
                raw_e, z_e, pr_v, ph_v, cv = float(mc[gi]), float((mc[gi] - mc.mean()) / (mc.std() + 1e-6)), float(gp[gi]), float(gph[gi]), causal_gene_mat[gi]
            else:
                raw_e, z_e, pr_v, ph_v, cv = 0.0, 0.0, 0.0, 0.0, np.zeros(64, np.float32)
            tstat_l.append([raw_e, z_e, pr_v, ph_v, float(np.log1p(np.linalg.norm(ridge_preds[idx])))])
            causal_l.append(cv)
        return tuple(np.asarray(x, np.float32) for x in [ctrl_l, teach_l, ridge_preds, tstat_l, causal_l])

    Xc, Xt, Xr, Xs, Xca = build_feats(tr_samples, pred_ridge_tr)
    ev_samples = [("RPE1", k, mu_rpe1, d_rpe1[k]) for k in ev_keys]
    Ec, Et, Er, Es, Eca = build_feats(ev_samples, pred_ridge_ev)

    # 常驻 GPU 训练 5-Member Deep Ensemble
    Xc_g, Xt_g, Xr_g, Xs_g, Xca_g, Y_g = [torch.tensor(a, device=DEVICE) for a in [Xc, Xt, Xr, Xs, Xca, Y_tr_lin]]
    Ec_g, Et_g, Er_g, Es_g, Eca_g = [torch.tensor(a, device=DEVICE) for a in [Ec, Et, Er, Es, Eca]]
    n_tr = Xc_g.shape[0]

    ens_preds = []
    seeds = [42, 101, 202, 303, 404]
    for sd in seeds:
        torch.manual_seed(sd)
        net = LineageInvariantPathwayNet(n_genes, Xt.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        net.train()
        for ep in range(45):
            perm = torch.randperm(n_tr, device=DEVICE)
            for i in range(0, n_tr, 256):
                idx = perm[i : i + 256]
                opt.zero_grad(set_to_none=True)
                pred = net(Xc_g[idx], Xt_g[idx], Xr_g[idx], Xs_g[idx], Xca_g[idx])
                loss = F.mse_loss(pred, Y_g[idx]) + 0.05 * (1.0 - F.cosine_similarity(pred, Y_g[idx], dim=-1)).mean()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            ens_preds.append(net(Ec_g, Et_g, Er_g, Es_g, Eca_g).cpu().numpy())

    ens_stack = np.stack(ens_preds, axis=0)  # (5, 2016, 3603)
    pred_mean = ens_stack.mean(axis=0)
    # 认知不确定性 1: Deep Ensemble 方差 (per-perturbation mean std)
    ens_std = ens_stack.std(axis=0).mean(axis=1)
    # 认知不确定性 2: 基因-教师-因果空间到训练集的 OOD 余弦距离
    tr_emb = np.hstack([Xt, Xca])
    ev_emb = np.hstack([Et, Eca])
    tr_emb_n = tr_emb / np.maximum(np.linalg.norm(tr_emb, axis=1, keepdims=True), 1e-6)
    ev_emb_n = ev_emb / np.maximum(np.linalg.norm(ev_emb, axis=1, keepdims=True), 1e-6)
    max_cos_sim = (ev_emb_n @ tr_emb_n.T).max(axis=1)
    ood_dist = 1.0 - max_cos_sim
    # 综合 UQ 分数 (标准化 Ensemble Std + OOD Dist)
    z_std = (ens_std - ens_std.mean()) / (ens_std.std() + 1e-6)
    z_ood = (ood_dist - ood_dist.mean()) / (ood_dist.std() + 1e-6)
    uq_score = 0.7 * z_std + 0.3 * z_ood

    # 评估 Ensemble 整体精度
    df_ev = evaluate_perturbation_predictions(d_rpe1, {k: pred_mean[i] for i, k in enumerate(ev_keys)}, "Exp34_5Ensemble")
    y_true_mat = np.vstack([d_rpe1[k] for k in ev_keys])
    mse_per_pert = np.mean((y_true_mat - pred_mean) ** 2, axis=1)
    p_all_per_pert = df_ev["pearson_delta_all"].values
    p_top20_per_pert = df_ev["pearson_delta_top20"].values

    r_std_mse, p_std_mse = spearmanr(ens_std, mse_per_pert)
    r_uq_mse, p_uq_mse = spearmanr(uq_score, mse_per_pert)
    r_ood_mse, p_ood_mse = spearmanr(ood_dist, mse_per_pert)

    # Risk-Coverage 选择性预测表 (保留最自信的 100%, 80%, 50%, 25% 扰动)
    order = np.argsort(uq_score)
    cov_rows = []
    for cov in [1.0, 0.8, 0.5, 0.25]:
        k_keep = max(10, int(cov * len(order)))
        sel = order[:k_keep]
        cov_rows.append({
            "coverage": cov,
            "n_perts": k_keep,
            "mean_mse": float(mse_per_pert[sel].mean()),
            "mean_pearson": float(np.nanmean(p_all_per_pert[sel])),
            "top20_pearson": float(np.nanmean(p_top20_per_pert[sel])),
        })
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    det_df = pd.DataFrame({
        "perturbation": ev_keys,
        "ensemble_std": ens_std,
        "ood_dist": ood_dist,
        "uq_score": uq_score,
        "mse": mse_per_pert,
        "pearson_all": p_all_per_pert,
        "pearson_top20": p_top20_per_pert,
    })
    det_df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    print(f"\n=== Exp34 5-Member Deep Ensemble 整体盲测精度 ({time.time()-t0:.1f}s) ===")
    print(f"  mean_pearson={np.nanmean(p_all_per_pert):.6f} | top20_pearson={np.nanmean(p_top20_per_pert):.6f} | mse={mse_per_pert.mean():.6f}")
    print(f"=== UQ 不确定性校准 Spearman 相关性 ===")
    print(f"  Ensemble_Std vs MSE : r = {r_std_mse:+.4f} (p = {p_std_mse:.2e})")
    print(f"  OOD_Distance vs MSE : r = {r_ood_mse:+.4f} (p = {p_ood_mse:.2e})")
    print(f"  Combined_UQ  vs MSE : r = {r_uq_mse:+.4f} (p = {p_uq_mse:.2e})")
    print("\n=== Risk-Coverage 选择性置信度过滤曲线 ===")
    print(cov_df.to_string(index=False))

    # 绘图：Risk-Coverage 曲线 + UQ 分位箱线校准图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    axes[0].plot(cov_df["coverage"] * 100, cov_df["mean_mse"], marker="o", linewidth=2.2, color="#c0392b", label="Mean MSE")
    axes[0].set_xlabel("Coverage (% Retained Confident Perturbations)", fontsize=9.5)
    axes[0].set_ylabel("Prediction MSE (Lower = Better)", fontsize=9.5, color="#c0392b")
    axes[0].invert_xaxis()
    axes[0].set_title(f"Risk-Coverage Selective Prediction\n(UQ vs MSE Spearman r={r_uq_mse:+.3f}, p={p_uq_mse:.1e})", fontsize=10, fontweight="bold")
    ax0_r = axes[0].twinx()
    ax0_r.plot(cov_df["coverage"] * 100, cov_df["top20_pearson"], marker="s", linewidth=2.0, color="#2980b9", label="Top-20 Pearson")
    ax0_r.set_ylabel("Top-20 DEG Pearson (Higher = Better)", fontsize=9.5, color="#2980b9")

    # 5 分位 UQ 校准箱线/柱状图
    det_df["uq_bin"] = pd.qcut(det_df["uq_score"], 5, labels=["Q1\n(Lowest UQ)", "Q2", "Q3", "Q4", "Q5\n(Highest UQ)"])
    bin_stats = det_df.groupby("uq_bin", observed=False)["mse"].mean()
    axes[1].bar(range(5), bin_stats.values, color=["#27ae60", "#52be80", "#f4d03f", "#e67e22", "#c0392b"], edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(bin_stats.index, fontsize=8.5)
    axes[1].set_ylabel("Observed Mean MSE on Unseen RPE1", fontsize=9.5)
    axes[1].set_title("Uncertainty Calibration by Quantile\n(Higher Predicted UQ -> Higher True Error)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved UQ calibration plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
