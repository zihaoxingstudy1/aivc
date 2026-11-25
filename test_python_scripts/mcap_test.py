# -*- coding: utf-8 -*-
"""
mcap.py 测试脚本：合并 Replogle K562_essential 与 RPE1 并生成跨细胞系基准数据集
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from mcap import merge_and_standardize_contexts
from ops import split_ood_perturbations
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    paths_str = r"D:\data\AIVC\data\ReplogleWeissman2022_K562_essential.h5ad;D:\data\AIVC\data\ReplogleWeissman2022_rpe1.h5ad"
    ctx_str = "K562;RPE1"
    merged_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000.h5ad"
    split_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"

    adata, _ = merge_and_standardize_contexts(
        h5ad_paths_str=paths_str,
        context_names_str=ctx_str,
        context_col="cell_line",
        pert_col="perturbation",
        control_label="control",
        min_cells_per_gene=100,
        min_cells_per_pert=30,
        target_sum=10000.0,
        n_top_genes=2000,
        keep_shared_pert_genes=True,
        save_h5ad_path=merged_h5ad,
        summary_csv_path=r"D:\data\AIVC\test_results\replogle_context_summary.csv",
    )

    # 跨细胞系 Zero-shot 划分：Train = K562, Test = RPE1
    adata_split, _ = split_ood_perturbations(
        adata_or_path=adata,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_mode="unseen_context",
        context_col="cell_line",
        val_contexts="",
        test_contexts="RPE1",
        seed=42,
        split_col="ood_split",
        subgroup_col="ood_subgroup",
        save_h5ad_path=split_h5ad,
        split_summary_csv=r"D:\data\AIVC\test_results\replogle_ood_split_summary.csv",
    )

    # 跑通跨细胞系线性基线评测
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
        per_pert_csv_path=r"D:\data\AIVC\test_results\replogle_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\replogle_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\replogle_baseline_benchmark.png",
    )
    print("mcap_test.py 跨细胞系基准全流程通过！")

# 中文注释：合并 Replogle K562 与 RPE1 数据集，执行跨细胞系 Zero-shot 划分并输出线性基线评测基准。
