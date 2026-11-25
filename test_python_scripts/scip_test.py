# -*- coding: utf-8 -*-
"""
scip.py 测试脚本：标准化 sci-Plex3 数据集并跑通跨细胞系 (A549+K562 -> MCF7) 药物响应线性基线
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from scip import prepare_drug_perturb_data
from ops import split_ood_perturbations
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    raw_h5ad = r"D:\data\AIVC\data\SrivatsanTrapnell2020_sciplex3.h5ad"
    out_h5ad = r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad"
    split_h5ad = r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad"

    adata, _ = prepare_drug_perturb_data(
        h5ad_path=raw_h5ad,
        pert_col="perturbation",
        control_label="control",
        cell_line_col="cell_line",
        valid_cell_lines_str="MCF7;A549;K562",
        dose_col="dose_value",
        time_col="time",
        target_time_val=24.0,
        pathway_col="pathway_level_1",
        min_cells_per_gene=200,
        min_cells_per_group=15,
        target_sum=10000.0,
        n_top_genes=2000,
        save_h5ad_path=out_h5ad,
        summary_csv_path=r"D:\data\AIVC\test_results\sciplex3_context_summary.csv",
    )

    # 跨细胞系 Zero-shot 划分：Train = A549 + K562, Test = MCF7
    adata_split, _ = split_ood_perturbations(
        adata_or_path=adata,
        pert_col="pert_dose",
        control_label="control",
        sep_char="||",
        split_mode="unseen_context",
        context_col="cell_line",
        val_contexts="",
        test_contexts="MCF7",
        seed=42,
        split_col="ood_split",
        subgroup_col="ood_subgroup",
        save_h5ad_path=split_h5ad,
        split_summary_csv=r"D:\data\AIVC\test_results\sciplex3_ood_split_summary.csv",
    )

    run_linear_baseline_benchmark(
        adata_or_path=adata_split,
        pert_col="pert_dose",
        control_label="control",
        sep_char="||",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=r"D:\data\AIVC\test_results\sciplex3_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\sciplex3_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\sciplex3_baseline_benchmark.png",
    )
    print("scip_test.py 小分子药物跨细胞系基准全流程通过！")

# 中文注释：标准化 sci-Plex3 药物扰动数据，执行 A549+K562 到 MCF7 的跨细胞系划分并运行线性基线评测。
