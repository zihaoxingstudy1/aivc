# -*- coding: utf-8 -*-
"""
ops.py 函数直接测试脚本
在预处理后的 Norman2019_hvg2000.h5ad 上测试 unseen_combo OOD 划分与防泄漏校验。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from ops import split_ood_perturbations


if __name__ == "__main__":
    in_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000.h5ad"
    out_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad"
    split_csv = r"D:\data\AIVC\test_results\norman2019_ood_split_summary.csv"

    adata, split_df = split_ood_perturbations(
        adata_or_path=in_h5ad,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_mode="unseen_combo",
        val_ratio=0.15,
        test_ratio=0.20,
        seed=42,
        split_col="ood_split",
        subgroup_col="ood_subgroup",
        save_h5ad_path=out_h5ad,
        split_summary_csv=split_csv,
    )

    assert os.path.exists(out_h5ad), "未生成带划分标签的 h5ad"
    assert os.path.exists(split_csv), "未生成划分统计表"
    assert set(adata.obs["ood_split"].unique()) == {"train", "val", "test"}, "划分集合不完整"
    print("ops_test.py 测试全部通过！")

# 中文注释：调用 ops 模块对 Norman2019_hvg2000.h5ad 执行 unseen_combo 划分测试并验证防泄漏与文件输出。
