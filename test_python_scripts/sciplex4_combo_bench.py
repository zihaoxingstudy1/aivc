# -*- coding: utf-8 -*-
"""
sci-Plex4 双药联合协同扰动数据集清洗、标准化与未见双药组合 (Unseen Drug Combo) 基准测试
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from ops import split_ood_perturbations
from lbb import run_linear_baseline_benchmark


if __name__ == "__main__":
    raw_h5 = r"D:\data\AIVC\data\SrivatsanTrapnell2020_sciplex4.h5ad"
    std_h5 = r"D:\data\AIVC\data\Sciplex4_hvg2000.h5ad"
    split_h5 = r"D:\data\AIVC\data\Sciplex4_hvg2000_split.h5ad"

    print("[sciPlex4] 正在读取数据...")
    adata = sc.read_h5ad(raw_h5)
    adata.obs["cell_line"] = adata.obs["cell_line"].astype(str)
    adata = adata[adata.obs["cell_line"].isin(["A549", "MCF7"])].copy()

    # 构造规范化单药/双药组合标签
    p1 = adata.obs["perturbation"].astype(str).values
    p2 = adata.obs["perturbation_2"].astype(str).values
    combo_labels = []
    for a, b in zip(p1, p2):
        toks = []
        if a != "control" and a != "nan":
            toks.append(a.strip())
        if b != "control" and b != "nan":
            toks.append(b.strip())
        if len(toks) == 0:
            combo_labels.append("control")
        else:
            combo_labels.append("_".join(sorted(toks)))
    adata.obs["pert_combo"] = combo_labels

    # 过滤样本过少组与低表达基因
    vc = adata.obs["pert_combo"].value_counts()
    valid_p = set(vc[(vc >= 30) | (vc.index == "control")].index)
    adata = adata[adata.obs["pert_combo"].isin(valid_p)].copy()

    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsr()
    else:
        adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))

    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_total(adata, target_sum=10000.0)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="cell_line", subset=True)
    print(f"[sciPlex4] 预处理后维度: {adata.shape}, 唯一药物组合数: {adata.obs['pert_combo'].nunique()}")
    adata.write_h5ad(std_h5)

    adata_split, _ = split_ood_perturbations(
        adata_or_path=adata,
        pert_col="pert_combo",
        control_label="control",
        sep_char="_",
        split_mode="unseen_combo",
        val_ratio=0.15,
        test_ratio=0.25,
        seed=42,
        save_h5ad_path=split_h5,
        split_summary_csv=r"D:\data\AIVC\test_results\sciplex4_ood_split_summary.csv",
    )

    run_linear_baseline_benchmark(
        adata_or_path=adata_split,
        pert_col="pert_combo",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=r"D:\data\AIVC\test_results\sciplex4_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\sciplex4_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\sciplex4_baseline_benchmark.png",
    )
    print("sciplex4_combo_bench.py 完成！")
