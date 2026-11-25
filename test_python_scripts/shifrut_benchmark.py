# -*- coding: utf-8 -*-
"""
ShifrutMarson2018 (原代人类 T 细胞免疫检查点筛选) 基准构建脚本
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from dlsp import load_and_standardize_perturb_data
from ops import split_ood_perturbations
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    raw_h5ad = r"D:\data\AIVC\data\ShifrutMarson2018.h5ad"
    std_h5ad = r"D:\data\AIVC\data\Shifrut2018_hvg2000.h5ad"
    split_h5ad = r"D:\data\AIVC\data\Shifrut2018_hvg2000_split.h5ad"

    adata, _ = load_and_standardize_perturb_data(
        h5ad_path=raw_h5ad,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        quality_col="",
        min_cells_per_gene=20,
        min_cells_per_pert=30,
        target_sum=10000.0,
        n_top_genes=2000,
        keep_pert_genes=True,
        save_h5ad_path=std_h5ad,
        summary_csv_path=r"D:\data\AIVC\test_results\shifrut2018_pert_summary.csv",
    )

    adata_split, _ = split_ood_perturbations(
        adata_or_path=adata,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_mode="unseen_pert",
        val_ratio=0.15,
        test_ratio=0.25,
        seed=42,
        split_col="ood_split",
        subgroup_col="ood_subgroup",
        save_h5ad_path=split_h5ad,
        split_summary_csv=r"D:\data\AIVC\test_results\shifrut2018_ood_split_summary.csv",
    )

    run_linear_baseline_benchmark(
        adata_or_path=adata_split,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=r"D:\data\AIVC\test_results\shifrut2018_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\shifrut2018_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\shifrut2018_baseline_benchmark.png",
    )
    print("shifrut_benchmark.py 原代 T 细胞基准构建完成！")
