# -*- coding: utf-8 -*-
"""
Norman 2019 夜间调优与两条腿消融对比脚本：
1. BioFlowCell-v2 (纯自研流匹配调优版)
2. DistillCell-GF104M-v2 (Geneformer 蒸馏幅度校准版)
3. BioFlowCell-Distill-v2 (融合 Geneformer 先验 + 原创流匹配)
"""

import os
import sys
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from bfc import train_and_eval_bioflowcell
from gf_distill import train_and_eval_distill_student, extract_geneformer_gene_embeddings


if __name__ == "__main__":
    split_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad"
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-104M_CLcancer"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"

    adata = sc.read_h5ad(split_h5ad)
    res_summaries = []

    # 1. 腿 1 调优版: BioFlowCell-v2 (纯自研)
    s_bfc, _ = train_and_eval_bioflowcell(
        adata_or_path=adata,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        prior_dim=64,
        hidden_dim=256,
        n_blocks=3,
        dropout=0.25,
        ridge_alpha=10.0,
        lr=3e-4,
        weight_decay=1e-3,
        epochs=45,
        cells_per_group_batch=32,
        groups_per_step=12,
        lambda_cfm=0.25,
        lambda_cos=0.8,
        lambda_huber=1.0,
        lambda_mse=25.0,
        ode_steps=8,
        seed=42,
        model_name="BioFlowCell-v2(Scratch)",
        top_k_str="20;50",
        save_ckpt_path=r"D:\data\AIVC\test_results\norman2019_bfc_v2.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_bfc_v2_per_pert.csv",
    )
    res_summaries.append(s_bfc)

    # 2. 腿 2 调优版: DistillCell-GF104M-v2 (幅度校准)
    s_gf, _, gf_gene_embs = train_and_eval_distill_student(
        adata_or_path=adata,
        geneformer_dir=gf_dir,
        token_dict_pkl=tok_pkl,
        gene_name_id_pkl=gn_pkl,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensemble_id",
        ridge_alpha=10.0,
        hidden_dim=256,
        epochs=100,
        lr=8e-4,
        lambda_distill=0.5,
        lambda_mse=25.0,
        top_k_str="20;50",
        model_name="DistillCell-GF104M-v2",
        save_ckpt_path=r"D:\data\AIVC\test_results\norman2019_distill_v2.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_distill_v2_per_pert.csv",
    )
    res_summaries.append(s_gf)

    # 3. 融合版: 把 Geneformer 768 维投影降维至 64 维后注入 BioFlowCell
    keys = sorted(gf_gene_embs.keys())
    mat = np.vstack([gf_gene_embs[k] for k in keys])
    pca = PCA(n_components=min(64, len(keys) - 1), random_state=42)
    mat_64 = pca.fit_transform(mat).astype(np.float32)
    mat_64 = mat_64 / (np.linalg.norm(mat_64, axis=1, keepdims=True) + 1e-6)
    gf_64_dict = {k: mat_64[i] for i, k in enumerate(keys)}

    s_fusion, _ = train_and_eval_bioflowcell(
        adata_or_path=adata,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        prior_dim=64,
        hidden_dim=256,
        n_blocks=3,
        dropout=0.25,
        ridge_alpha=10.0,
        lr=3e-4,
        weight_decay=1e-3,
        epochs=45,
        cells_per_group_batch=32,
        groups_per_step=12,
        lambda_cfm=0.25,
        lambda_cos=0.8,
        lambda_huber=1.0,
        lambda_mse=25.0,
        ode_steps=8,
        seed=42,
        teacher_gene_emb_dict=gf_64_dict,
        model_name="BioFlowCell-Distill(Fusion)",
        top_k_str="20;50",
        save_ckpt_path=r"D:\data\AIVC\test_results\norman2019_bfc_fusion.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_bfc_fusion_per_pert.csv",
    )
    res_summaries.append(s_fusion)

    # 汇总与线性基线合并保存
    base_df = pd.read_csv(r"D:\data\AIVC\test_results\norman2019_baseline_summary.csv")
    all_df = pd.concat([base_df] + res_summaries, ignore_index=True)
    all_df.to_csv(r"D:\data\AIVC\test_results\norman2019_all_models_summary.csv", index=False, encoding="utf-8")
    print("\n=== Norman 2019 全模型对比总表 ===")
    cols = ["model", "pearson_delta_all_mean", "pearson_delta_top20_mean", "dir_acc_top20_mean", "mse_all_mean"]
    print(all_df[cols].to_string(index=False))

# 中文注释：在 Norman 2019 上执行 BioFlowCell-v2、DistillCell-GF104M-v2 与融合版 BioFlowCell-Distill 的同台消融对比。
