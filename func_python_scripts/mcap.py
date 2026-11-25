# -*- coding: utf-8 -*-
"""
多语境单细胞扰动数据集对齐合并与标准化模块 (Merge Context AnnData Perturbation)
用于将多个细胞系/批次的 .h5ad（如 Replogle K562 与 RPE1）按共有基因对齐合并并标准化。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse


def merge_and_standardize_contexts(
    h5ad_paths_str: str,
    context_names_str: str,
    context_col: str = "cell_line",
    pert_col: str = "perturbation",
    control_label: str = "control",
    min_cells_per_gene: int = 50,
    min_cells_per_pert: int = 30,
    target_sum: float = 10000.0,
    n_top_genes: int = 2000,
    keep_shared_pert_genes: bool = True,
    save_h5ad_path: str = "",
    summary_csv_path: str = "",
):
    """
    读取多个单细胞扰动 .h5ad 文件，按共有基因拼接，执行质控、标准化与 HVG 筛选。

    Args:
        h5ad_paths_str: 分号分隔的 .h5ad 文件路径列表。
        context_names_str: 分号分隔的语境标签列表（与路径一一对应，如 'RPE1;K562'）。
        context_col: 写入 obs 的语境列名。
        pert_col: 扰动列名。
        control_label: 对照组标签。
        min_cells_per_gene: 基因至少在多少个细胞中表达才保留。
        min_cells_per_pert: 每个 (context, pert) 组至少保留的细胞数。
        target_sum: normalize_total 目标文库大小。
        n_top_genes: HVG 数量。
        keep_shared_pert_genes: 是否保留出现在扰动标签中的共有基因。
        save_h5ad_path: 合并后 .h5ad 保存路径。
        summary_csv_path: 合并统计摘要 CSV 路径。

    Returns:
        merged_adata: 合并标准化后的 AnnData。
        summary_df: 各语境统计摘要 DataFrame。
    """
    paths = [p.strip() for p in str(h5ad_paths_str).split(";") if p.strip()]
    ctx_names = [c.strip() for c in str(context_names_str).split(";") if c.strip()]
    if len(paths) != len(ctx_names) or len(paths) == 0:
        raise ValueError("h5ad_paths_str 与 context_names_str 数量必须一致且非空！")

    context_col = str(context_col)
    pert_col = str(pert_col)
    control_label = str(control_label)
    min_cells_per_gene = int(min_cells_per_gene)
    min_cells_per_pert = int(min_cells_per_pert)
    target_sum = float(target_sum)
    n_top_genes = int(n_top_genes)
    keep_shared_pert_genes = bool(keep_shared_pert_genes)

    adatas = []
    common_genes = None
    for p, cname in zip(paths, ctx_names):
        print(f"[MCAP] 正在读取语境 '{cname}': {p}")
        a = sc.read_h5ad(p)
        a.var_names_make_unique()
        a.obs_names_make_unique()
        a.obs[context_col] = str(cname)
        a.obs[pert_col] = a.obs[pert_col].astype(str)
        # 仅保留必要 obs 列以大幅节省内存与加速拼接
        keep_cols = [c for c in [context_col, pert_col, "nperts", "ncounts", "ngenes"] if c in a.obs.columns]
        a.obs = a.obs[keep_cols].copy()
        if sparse.issparse(a.X):
            a.X = a.X.astype(np.float32).tocsr()
        else:
            a.X = sparse.csr_matrix(np.asarray(a.X, dtype=np.float32))

        genes_set = set(a.var_names)
        common_genes = genes_set if common_genes is None else (common_genes & genes_set)
        adatas.append(a)

    common_genes = sorted(list(common_genes))
    print(f"[MCAP] 跨数据集共有基因数: {len(common_genes)}")
    adatas = [a[:, common_genes].copy() for a in adatas]

    merged = ad.concat(adatas, join="inner", label="dataset_src", keys=ctx_names, index_unique="-")
    del adatas
    print(f"[MCAP] 拼接后总维度: {merged.n_obs} 细胞 x {merged.n_vars} 基因")

    # 过滤每个语境下样本量过少的扰动组（保留对照组）
    keep_mask = np.zeros(merged.n_obs, dtype=bool)
    for cname in ctx_names:
        ctx_mask = (merged.obs[context_col] == cname).values
        vc = merged.obs.loc[ctx_mask, pert_col].value_counts()
        valid_p = set(vc[(vc >= min_cells_per_pert) | (vc.index == control_label)].index)
        keep_mask |= ctx_mask & merged.obs[pert_col].isin(valid_p).values
    merged = merged[keep_mask].copy()
    print(f"[MCAP] 扰动组过滤后: {merged.n_obs} 细胞")

    if min_cells_per_gene > 0:
        sc.pp.filter_genes(merged, min_cells=min_cells_per_gene)
        print(f"[MCAP] 基因过滤后: {merged.n_vars} 基因")

    sample_vals = merged.X[: min(500, merged.n_obs)].toarray()
    is_raw = bool(np.allclose(sample_vals, np.round(sample_vals)) and sample_vals.max() > 20.0)
    if is_raw:
        sc.pp.normalize_total(merged, target_sum=target_sum)
        sc.pp.log1p(merged)
        print(f"[MCAP] 已执行 normalize_total(target_sum={target_sum}) + log1p")

    if 0 < n_top_genes < merged.n_vars:
        sc.pp.highly_variable_genes(merged, n_top_genes=n_top_genes, batch_key=context_col, subset=False)
        hvg_mask = merged.var["highly_variable"].values.copy()
        if keep_shared_pert_genes:
            pert_set = set(merged.obs[pert_col].unique()) - {control_label}
            in_var = merged.var_names.isin(pert_set)
            hvg_mask = hvg_mask | in_var
        merged = merged[:, hvg_mask].copy()
        print(f"[MCAP] HVG 筛选后最终维度: {merged.n_obs} 细胞 x {merged.n_vars} 基因")

    rows = []
    for cname, grp in merged.obs.groupby(context_col):
        rows.append(
            {
                "context": cname,
                "n_cells": int(len(grp)),
                "n_ctrl_cells": int((grp[pert_col] == control_label).sum()),
                "n_unique_perts": int(grp.loc[grp[pert_col] != control_label, pert_col].nunique()),
            }
        )
    summary_df = pd.DataFrame(rows)
    print("[MCAP] 语境摘要:\n", summary_df.to_string(index=False))

    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    if save_h5ad_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_h5ad_path)), exist_ok=True)
        merged.write_h5ad(save_h5ad_path)
        print(f"[MCAP] 已保存至: {save_h5ad_path}")

    return merged, summary_df


# 中文注释：本模块实现多个单细胞扰动数据集的共有基因对齐、跨语境合并、标准化与高变基因筛选。
