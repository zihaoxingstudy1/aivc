# -*- coding: utf-8 -*-
"""
dlsp.py 函数直接测试脚本
测试真实数据集 NormanWeissman2019_filtered.h5ad 的加载、质控与标准化流程。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from dlsp import load_and_standardize_perturb_data


if __name__ == "__main__":
    raw_h5ad = r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad"
    out_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000.h5ad"
    summary_csv = r"D:\data\AIVC\test_results\norman2019_pert_summary.csv"

    adata, summary_df = load_and_standardize_perturb_data(
        h5ad_path=raw_h5ad,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        quality_col="good_coverage",
        min_cells_per_gene=20,
        min_cells_per_pert=30,
        target_sum=10000.0,
        n_top_genes=2000,
        keep_pert_genes=True,
        save_h5ad_path=out_h5ad,
        summary_csv_path=summary_csv,
    )

    print("\n=== 测试验证摘要 ===")
    print("输出 AnnData 维度:", adata.shape)
    print("总扰动组数:", len(summary_df))
    print("各 n_perts 组数量分布:\n", summary_df["n_perts"].value_counts().sort_index())
    x_sub = adata.X[:100].toarray() if hasattr(adata.X, "toarray") else adata.X[:100]
    print(f"标准化后抽样数值范围: min={x_sub.min():.4f}, max={x_sub.max():.4f}, mean={x_sub.mean():.4f}")
    assert adata.n_obs > 50000, "细胞数异常偏少"
    assert adata.n_vars >= 2000, "基因数少于目标 HVG"
    assert os.path.exists(out_h5ad), "未生成预处理 h5ad"
    assert os.path.exists(summary_csv), "未生成摘要 csv"
    print("dlsp_test.py 测试全部通过！")

# 中文注释：调用 dlsp 模块对真实 Norman 2019 数据集执行加载与标准化测试，并验证输出文件完整性。
