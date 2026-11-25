# -*- coding: utf-8 -*-
"""
AIVC 跨 5 大场景 × 多技术路线（线性基线 vs 原创流匹配 vs 多教师大基座蒸馏）全自动评测总控脚本
覆盖：
1. Norman 2019 (双基因组合上位性)
2. Tian 2021 (iPSC 神经元未见单基因 Zero-shot)
3. Shifrut 2018 (原代人类 T 细胞免疫检查点 Zero-shot)
4. Replogle K562 -> RPE1 (跨细胞系基因扰动 Zero-shot)
5. sci-Plex3 A549+K562 -> MCF7 (跨细胞系小分子药物 + 连续剂量响应)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from bfc import train_and_eval_bioflowcell
from gf_distill import train_and_eval_distill_student


GF104_DIR = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-104M_CLcancer"
GF316_DIR = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
TOK_PKL = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
GN_PKL = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"


def run_gene_datasets():
    all_records = []

    # --- 1. Norman 2019: 补充 Geneformer-V2-316M 蒸馏，并与现有总表合并 ---
    print("\n" + "=" * 60)
    print(">>> 场景 1: Norman 2019 (补充 Geneformer-V2-316M 蒸馏)")
    print("=" * 60)
    s_norman_316, _, _ = train_and_eval_distill_student(
        adata_or_path=r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad",
        geneformer_dir=GF316_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
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
        model_name="DistillCell-GF316M",
        save_ckpt_path=r"D:\data\AIVC\test_results\norman2019_distill_gf316m.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_distill_gf316m_per_pert.csv",
    )
    norman_all_path = r"D:\data\AIVC\test_results\norman2019_all_models_summary.csv"
    if os.path.exists(norman_all_path):
        df_n = pd.concat([pd.read_csv(norman_all_path), s_norman_316], ignore_index=True)
    else:
        df_n = s_norman_316
    df_n.to_csv(norman_all_path, index=False, encoding="utf-8")
    df_n["dataset"] = "Norman2019 (Combo)"
    all_records.append(df_n)

    # --- 2. Tian 2021: iPSC 神经元未见单基因 Zero-shot ---
    print("\n" + "=" * 60)
    print(">>> 场景 2: Tian 2021 iPSC 神经元 (Unseen Gene Zero-shot)")
    print("=" * 60)
    tian_h5 = r"D:\data\AIVC\data\Tian2021_hvg2000_split.h5ad"
    s_t_bfc, _ = train_and_eval_bioflowcell(
        adata_or_path=tian_h5,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        epochs=40,
        lr=3e-4,
        dropout=0.25,
        weight_decay=1e-3,
        lambda_mse=25.0,
        model_name="BioFlowCell-v2(Scratch)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\tian2021_bfc_per_pert.csv",
    )
    s_t_104, _, _ = train_and_eval_distill_student(
        adata_or_path=tian_h5,
        geneformer_dir=GF104_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensembl_id",
        epochs=100,
        lr=6e-4,
        lambda_mse=25.0,
        model_name="DistillCell-GF104M",
        per_pert_csv_path=r"D:\data\AIVC\test_results\tian2021_gf104_per_pert.csv",
    )
    s_t_316, _, _ = train_and_eval_distill_student(
        adata_or_path=tian_h5,
        geneformer_dir=GF316_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensembl_id",
        epochs=100,
        lr=6e-4,
        lambda_mse=25.0,
        model_name="DistillCell-GF316M",
        per_pert_csv_path=r"D:\data\AIVC\test_results\tian2021_gf316_per_pert.csv",
    )
    df_t = pd.concat([pd.read_csv(r"D:\data\AIVC\test_results\tian2021_baseline_summary.csv"), s_t_bfc, s_t_104, s_t_316], ignore_index=True)
    df_t.to_csv(r"D:\data\AIVC\test_results\tian2021_all_models_summary.csv", index=False, encoding="utf-8")
    df_t["dataset"] = "Tian2021 (iPSC Neuron)"
    all_records.append(df_t)

    # --- 3. Shifrut 2018: 原代人类 T 细胞 ---
    print("\n" + "=" * 60)
    print(">>> 场景 3: Shifrut 2018 原代人类 T 细胞 (Unseen Gene Zero-shot)")
    print("=" * 60)
    shif_h5 = r"D:\data\AIVC\data\Shifrut2018_hvg2000_split.h5ad"
    s_s_bfc, _ = train_and_eval_bioflowcell(
        adata_or_path=shif_h5,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        epochs=40,
        lr=3e-4,
        dropout=0.25,
        weight_decay=1e-3,
        lambda_mse=25.0,
        model_name="BioFlowCell-v2(Scratch)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\shifrut2018_bfc_per_pert.csv",
    )
    s_s_104, _, _ = train_and_eval_distill_student(
        adata_or_path=shif_h5,
        geneformer_dir=GF104_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensembl_id",
        epochs=100,
        lr=6e-4,
        lambda_mse=25.0,
        model_name="DistillCell-GF104M",
        per_pert_csv_path=r"D:\data\AIVC\test_results\shifrut2018_gf104_per_pert.csv",
    )
    s_s_316, _, _ = train_and_eval_distill_student(
        adata_or_path=shif_h5,
        geneformer_dir=GF316_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensembl_id",
        epochs=100,
        lr=6e-4,
        lambda_mse=25.0,
        model_name="DistillCell-GF316M",
        per_pert_csv_path=r"D:\data\AIVC\test_results\shifrut2018_gf316_per_pert.csv",
    )
    df_s = pd.concat([pd.read_csv(r"D:\data\AIVC\test_results\shifrut2018_baseline_summary.csv"), s_s_bfc, s_s_104, s_s_316], ignore_index=True)
    df_s.to_csv(r"D:\data\AIVC\test_results\shifrut2018_all_models_summary.csv", index=False, encoding="utf-8")
    df_s["dataset"] = "Shifrut2018 (Primary T)"
    all_records.append(df_s)

    # --- 4. Replogle K562 -> RPE1: 跨细胞系 Zero-shot ---
    print("\n" + "=" * 60)
    print(">>> 场景 4: Replogle K562 -> RPE1 (跨细胞系 Zero-shot)")
    print("=" * 60)
    rep_h5 = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    s_r_bfc, _ = train_and_eval_bioflowcell(
        adata_or_path=rep_h5,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        context_col="cell_line",
        epochs=15,
        cells_per_group_batch=16,
        groups_per_step=32,
        lr=3e-4,
        dropout=0.25,
        weight_decay=1e-3,
        lambda_mse=25.0,
        model_name="BioFlowCell-v2(Scratch)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\replogle_bfc_per_pert.csv",
    )
    s_r_104, _, _ = train_and_eval_distill_student(
        adata_or_path=rep_h5,
        geneformer_dir=GF104_DIR,
        token_dict_pkl=TOK_PKL,
        gene_name_id_pkl=GN_PKL,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        context_col="cell_line",
        ensembl_col="ensembl_id",
        epochs=80,
        lr=6e-4,
        lambda_mse=25.0,
        model_name="DistillCell-GF104M",
        per_pert_csv_path=r"D:\data\AIVC\test_results\replogle_gf104_per_pert.csv",
    )
    df_r = pd.concat([pd.read_csv(r"D:\data\AIVC\test_results\replogle_baseline_summary.csv"), s_r_bfc, s_r_104], ignore_index=True)
    df_r.to_csv(r"D:\data\AIVC\test_results\replogle_all_models_summary.csv", index=False, encoding="utf-8")
    df_r["dataset"] = "Replogle (K562->RPE1)"
    all_records.append(df_r)

    # --- 5. sci-Plex3 A549+K562 -> MCF7: 跨细胞系小分子药物 + 剂量 ---
    print("\n" + "=" * 60)
    print(">>> 场景 5: sci-Plex3 A549+K562 -> MCF7 (跨细胞系小分子药物)")
    print("=" * 60)
    sci_h5 = r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad"
    s_sci_bfc, _ = train_and_eval_bioflowcell(
        adata_or_path=sci_h5,
        pert_col="pert_dose",
        control_label="control",
        sep_char="||",
        split_col="ood_split",
        context_col="cell_line",
        dose_col="log10_dose",
        epochs=15,
        cells_per_group_batch=16,
        groups_per_step=24,
        lr=3e-4,
        dropout=0.25,
        weight_decay=1e-3,
        lambda_mse=25.0,
        model_name="BioFlowCell-v2(Drug)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\sciplex3_bfc_per_pert.csv",
    )
    df_sci = pd.concat([pd.read_csv(r"D:\data\AIVC\test_results\sciplex3_baseline_summary.csv"), s_sci_bfc], ignore_index=True)
    df_sci.to_csv(r"D:\data\AIVC\test_results\sciplex3_all_models_summary.csv", index=False, encoding="utf-8")
    df_sci["dataset"] = "sci-Plex3 (Drug Cross-CL)"
    all_records.append(df_sci)

    # --- 6. 汇总全 5 大场景总表与总对比图 ---
    master_df = pd.concat(all_records, ignore_index=True)
    master_csv = r"D:\data\AIVC\test_results\all_5_datasets_benchmark_summary.csv"
    master_df.to_csv(master_csv, index=False, encoding="utf-8")
    print("\n" + "#" * 60)
    print(">>> 全 5 大场景对比汇总表:")
    cols = ["dataset", "model", "pearson_delta_all_mean", "pearson_delta_top20_mean", "mse_all_mean"]
    print(master_df[cols].to_string(index=False))

    # 绘图
    plt.figure(figsize=(14, 7), dpi=180)
    sns.set_style("whitegrid")
    ax = sns.barplot(data=master_df, x="dataset", y="pearson_delta_all_mean", hue="model")
    ax.set_title("AIVC Multi-Scenario Benchmark: Linear Baseline vs BioFlowCell vs Multi-Teacher Distillation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Pearson(Delta) - All Genes", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(r"D:\data\AIVC\test_results\all_5_datasets_comparison.png", bbox_inches="tight")
    plt.close()
    print(">>> 全部 5 大场景对比实验与图表生成完成！")


if __name__ == "__main__":
    run_gene_datasets()
