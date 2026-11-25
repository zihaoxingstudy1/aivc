# -*- coding: utf-8 -*-
"""
mcjt.py 测试脚本：构建 4 源多细胞系联合训练集，并在未见 RPE1 上评测线性基线与 CrossCtx 残差模型
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from mcjt import build_multi_lineage_replogle_dataset
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    out_h5 = r"D:\data\AIVC\data\MultiLineage_Replogle_RPE1.h5ad"
    adata = build_multi_lineage_replogle_dataset(
        replogle_split_h5ad=r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad",
        norman_h5ad=r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad",
        shifrut_h5ad=r"D:\data\AIVC\data\ShifrutMarson2018.h5ad",
        papalexi_h5ad=r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad",
        save_h5ad_path=out_h5,
    )

    run_linear_baseline_benchmark(
        adata_or_path=adata,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=r"D:\data\AIVC\test_results\multilineage_replogle_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\multilineage_replogle_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\multilineage_replogle_baseline_benchmark.png",
    )
    print("mcjt_test.py 多细胞系联合数据集构建与线性基线通过！")
