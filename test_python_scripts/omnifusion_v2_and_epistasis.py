# -*- coding: utf-8 -*-
"""
AIVC-OmniFusion-v2 调优与不确定性量化 (MC-Dropout UQ) + 非线性协同归因分析脚本
目标：
1. 将 Norman 2019 上的 MSE 从 0.0056 压低至 <= 0.0022，同时保持 >0.885 Pearson 与 >99% DirAcc；
2. 保持 Shifrut 2018 (原代 T 细胞) 全场第一；
3. 计算每个测试扰动的 MC-Dropout 预测不确定性 (Epistemic Uncertainty)，验证 Uncertainty vs Error 正相关性；
4. 导出经典非线性协同组合 (如 BCL2L11_BAK1) 的基因级预测对比与 Top DEG 归因图。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from omni_fusion import train_and_eval_omni_fusion, GatedMultiTeacherFusionNet
from gf_distill import extract_geneformer_gene_embeddings, extract_geneformer_cell_teacher_embeddings
from bfc import build_gene_coexp_prior
from lbb import _compute_group_deltas, _to_dense_array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


if __name__ == "__main__":
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")

    # 1. Norman 2019 精细调优 (提高 MSE 权重至 40.0, epochs=140, lr=8e-4, lambda_distill=0.2)
    adata_n = sc.read_h5ad(r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad")
    all_non_ctrl = sorted(set(adata_n.obs.loc[adata_n.obs["perturbation"] != "control", "perturbation"].astype(str)))
    vocab = sorted({t.strip() for p in all_non_ctrl for t in p.split("_") if t.strip()})
    var_ens = {g: str(adata_n.var["ensemble_id"].iloc[i]) for i, g in enumerate(adata_n.var_names)}

    gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)
    coexp_dict = build_gene_coexp_prior(adata_n[adata_n.obs["ood_split"] == "train"].X, list(adata_n.var_names), prior_dim=64)
    teacher_ctx = extract_geneformer_cell_teacher_embeddings(
        adata_n, gf_dir, tok_pkl, gn_pkl, ensembl_col="ensemble_id",
        max_seq_len=512, batch_size=32, max_cells_per_group=16,
        pert_col="perturbation", split_col="ood_split", target_split="train",
    )

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}

    s_v2, p_v2 = train_and_eval_omni_fusion(
        adata_or_path=adata_n,
        teacher_gene_dicts=teachers,
        teacher_cell_embs=teacher_ctx,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        bottleneck_dim=64,
        hidden_dim=192,
        dropout=0.20,
        epochs=140,
        lr=8e-4,
        weight_decay=3e-3,
        lambda_distill=0.2,
        lambda_mse=40.0,
        model_name="AIVC-OmniFusion-v2",
        save_ckpt_path=r"D:\data\AIVC\test_results\norman2019_omnifusion_v2.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_omnifusion_v2_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\norman2019_omnifusion_v2_summary.csv",
    )

    # 2. 对比线性基线 vs AIVC-OmniFusion-v2 在强非线性协同组合上的表现并作发表级对比图
    ridge_df = pd.read_csv(r"D:\data\AIVC\test_results\norman2019_baseline_per_pert.csv")
    ridge_df = ridge_df[ridge_df["model"].str.startswith("Linear-Ridge")].copy()
    merged_cmp = pd.merge(
        ridge_df[["perturbation", "pearson_delta_all", "pearson_delta_top20", "mse_all"]],
        p_v2[["perturbation", "pearson_delta_all", "pearson_delta_top20", "mse_all"]],
        on="perturbation",
        suffixes=("_ridge", "_omni"),
    )
    merged_cmp["gain_all"] = merged_cmp["pearson_delta_all_omni"] - merged_cmp["pearson_delta_all_ridge"]
    merged_cmp = merged_cmp.sort_values("gain_all", ascending=False)
    print("\n=== AIVC-OmniFusion-v2 相对 Linear-Ridge 提升最大的 Top 8 组合 ===")
    print(merged_cmp.head(8).round(4).to_string(index=False))
    merged_cmp.to_csv(r"D:\data\AIVC\test_results\norman2019_ridge_vs_omnifusion_v2.csv", index=False, encoding="utf-8")

    # 绘制 1x2 散点 + 增益对比图
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)

    ax = axes[0]
    ax.scatter(merged_cmp["pearson_delta_all_ridge"], merged_cmp["pearson_delta_all_omni"], c="#e74c3c", s=55, alpha=0.85, edgecolors="k", linewidth=0.5)
    lims = [-0.15, 1.02]
    ax.plot(lims, lims, "--", color="gray", linewidth=1.2, label="y = x (Equal Performance)")
    for _, row in merged_cmp.head(4).iterrows():
        ax.annotate(row["perturbation"], (row["pearson_delta_all_ridge"], row["pearson_delta_all_omni"]), xytext=(5, -5), textcoords="offset points", fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Linear-Ridge Pearson(Delta)", fontsize=10, fontweight="bold")
    ax.set_ylabel("AIVC-OmniFusion-v2 Pearson(Delta)", fontsize=10, fontweight="bold")
    ax.set_title("Per-Perturbation Gain on Unseen Combos (Norman 2019)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    top_gains = merged_cmp.head(10).iloc[::-1]
    ax2.barh(top_gains["perturbation"], top_gains["gain_all"], color="#2ecc71", edgecolor="k", linewidth=0.5)
    ax2.set_xlabel("Pearson(Delta) Gain over Linear-Ridge", fontsize=10, fontweight="bold")
    ax2.set_title("Top 10 Non-Linear Epistasis Combos Unlocked", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(r"D:\data\AIVC\test_results\norman2019_epistasis_unlock.png", bbox_inches="tight")
    plt.close(fig)
    print("已保存发表级非线性协同对比图: norman2019_epistasis_unlock.png")
