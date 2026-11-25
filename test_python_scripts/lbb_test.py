# -*- coding: utf-8 -*-
"""
lbb.py 函数直接测试脚本
在 Norman2019_hvg2000_split.h5ad 的 test 集上运行三大基线评测，生成结果表与对比图。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    split_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad"
    per_pert_csv = r"D:\data\AIVC\test_results\norman2019_baseline_per_pert.csv"
    summary_csv = r"D:\data\AIVC\test_results\norman2019_baseline_summary.csv"
    plot_png = r"D:\data\AIVC\test_results\norman2019_baseline_benchmark.png"

    summary_df, per_pert_df = run_linear_baseline_benchmark(
        adata_or_path=split_h5ad,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=per_pert_csv,
        summary_csv_path=summary_csv,
        plot_png_path=plot_png,
    )

    assert os.path.exists(per_pert_csv), "未生成逐扰动评测表"
    assert os.path.exists(summary_csv), "未生成汇总表"
    assert os.path.exists(plot_png), "未生成对比图"
    print("lbb_test.py 测试全部通过！")

# 中文注释：调用 lbb 模块对 Norman 2019 OOD 测试集执行三大基线评测与可视化输出。
