# -*- coding: utf-8 -*-
"""
小分子药物扰动数据集清洗、标准化与跨细胞系+剂量基线准备模块 (sci-Plex3 Perturbation)
处理包含多细胞系、多药物、多剂量与时间点的药物扰动 .h5ad 数据集。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def prepare_drug_perturb_data(
    h5ad_path: str,
    pert_col: str = "perturbation",
    control_label: str = "control",
    cell_line_col: str = "cell_line",
    valid_cell_lines_str: str = "MCF7;A549;K562",
    dose_col: str = "dose_value",
    time_col: str = "time",
    target_time_val: float = 24.0,
    pathway_col: str = "pathway_level_1",
    min_cells_per_gene: int = 200,
    min_cells_per_group: int = 15,
    target_sum: float = 10000.0,
    n_top_genes: int = 2000,
    save_h5ad_path: str = "",
    summary_csv_path: str = "",
):
    """
    加载并标准化小分子药物扰动数据，过滤缺失细胞系与低表达基因，构建药物-剂量标签。

    Args:
        h5ad_path: 输入 .h5ad 文件路径。
        pert_col: 药物名称列。
        control_label: 对照组名称。
        cell_line_col: 细胞系列名。
        valid_cell_lines_str: 分号分隔的有效细胞系列表。
        dose_col: 剂量数值列。
        time_col: 处理时间列（为空则不过滤时间）。
        target_time_val: 保留的目标时间点（如 24.0 小时）。
        pathway_col: 药物通路列。
        min_cells_per_gene: 基因至少在多少细胞中表达才保留。
        min_cells_per_group: 每个 (cell_line, perturbation, dose) 组最少细胞数。
        target_sum: normalize_total 目标值。
        n_top_genes: HVG 数量。
        save_h5ad_path: 输出标准化 .h5ad 路径。
        summary_csv_path: 输出汇总表路径。

    Returns:
        adata: 预处理后的 AnnData。
        summary_df: 细胞系与药物统计表。
    """
    h5ad_path = str(h5ad_path)
    pert_col = str(pert_col)
    control_label = str(control_label)
    cell_line_col = str(cell_line_col)
    valid_cls = [x.strip() for x in str(valid_cell_lines_str).split(";") if x.strip()]
    dose_col = str(dose_col)
    time_col = str(time_col) if time_col else ""
    target_time_val = float(target_time_val)
    pathway_col = str(pathway_col) if pathway_col else ""
    min_cells_per_gene = int(min_cells_per_gene)
    min_cells_per_group = int(min_cells_per_group)
    target_sum = float(target_sum)
    n_top_genes = int(n_top_genes)

    print(f"[SCIP] 正在读取药物扰动数据: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[SCIP] 原始维度: {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    # 1. 过滤有效细胞系与时间点
    adata.obs[cell_line_col] = adata.obs[cell_line_col].astype(str)
    mask = adata.obs[cell_line_col].isin(valid_cls).values
    if time_col and time_col in adata.obs.columns:
        time_vals = pd.to_numeric(adata.obs[time_col], errors="coerce")
        mask &= (time_vals == target_time_val).values

    adata = adata[mask].copy()
    print(f"[SCIP] 过滤细胞系 {valid_cls} & time={target_time_val} 后: {adata.n_obs} 细胞")

    # 2. 规范化 obs 列并构造 drug_dose 标签
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)
    adata.obs[dose_col] = pd.to_numeric(adata.obs[dose_col], errors="coerce").fillna(0.0).astype(np.float32)
    adata.obs["log10_dose"] = np.log10(adata.obs[dose_col].values + 1.0).astype(np.float32)
    if pathway_col and pathway_col in adata.obs.columns:
        adata.obs[pathway_col] = adata.obs[pathway_col].astype(str)

    pert_dose_labels = [
        control_label if p == control_label else f"{p}__dose_{int(round(d))}"
        for p, d in zip(adata.obs[pert_col], adata.obs[dose_col])
    ]
    adata.obs["pert_dose"] = pert_dose_labels

    # 过滤样本过少的 (cell_line, pert_dose) 组（保留对照组）
    grp_key = adata.obs[cell_line_col] + "||" + adata.obs["pert_dose"]
    vc = grp_key.value_counts()
    valid_keys = set(vc[vc >= min_cells_per_group].index)
    keep_grp_mask = grp_key.isin(valid_keys).values | (adata.obs[pert_col] == control_label).values
    adata = adata[keep_grp_mask].copy()
    print(f"[SCIP] 组样本量过滤后: {adata.n_obs} 细胞")

    # 3. 过滤低表达基因
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsr()
    else:
        adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))

    if min_cells_per_gene > 0:
        before_g = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        print(f"[SCIP] 基因过滤 (min_cells={min_cells_per_gene}): {before_g} -> {adata.n_vars} 基因")

    # 4. 标准化 + log1p
    sample_vals = adata.X[: min(500, adata.n_obs)].toarray()
    is_raw = bool(np.allclose(sample_vals, np.round(sample_vals)) and sample_vals.max() > 20.0)
    if is_raw:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print(f"[SCIP] 已执行 normalize_total(target_sum={target_sum}) + log1p")

    # 5. 高变基因筛选
    if 0 < n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=cell_line_col, subset=True)
        print(f"[SCIP] HVG 筛选后最终维度: {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    # 6. 汇总统计
    rows = []
    for cl, grp in adata.obs.groupby(cell_line_col):
        rows.append(
            {
                "cell_line": cl,
                "n_cells": int(len(grp)),
                "n_ctrl_cells": int((grp[pert_col] == control_label).sum()),
                "n_unique_drugs": int(grp.loc[grp[pert_col] != control_label, pert_col].nunique()),
                "n_unique_drug_doses": int(grp.loc[grp["pert_dose"] != control_label, "pert_dose"].nunique()),
            }
        )
    summary_df = pd.DataFrame(rows)
    print("[SCIP] 药物数据集摘要:\n", summary_df.to_string(index=False))

    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    if save_h5ad_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_h5ad_path)), exist_ok=True)
        adata.write_h5ad(save_h5ad_path)
        print(f"[SCIP] 已保存至: {save_h5ad_path}")

    return adata, summary_df


# 中文注释：本模块实现 sci-Plex3 等小分子药物扰动数据的细胞系过滤、时间筛选、药物-剂量标签构造、标准化及 HVG 筛选。
