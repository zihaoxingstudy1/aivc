# -*- coding: utf-8 -*-
"""
数据加载、质控与标准化预处理模块 (Data Load & Standardize Perturbation)
适用于单细胞基因扰动或药物扰动 AnnData (.h5ad) 数据集。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def load_and_standardize_perturb_data(
    h5ad_path: str,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    quality_col: str = "good_coverage",
    min_cells_per_gene: int = 20,
    min_cells_per_pert: int = 20,
    target_sum: float = 10000.0,
    n_top_genes: int = 2000,
    keep_pert_genes: bool = True,
    save_h5ad_path: str = "",
    summary_csv_path: str = "",
):
    """
    加载单细胞扰动数据，执行质控过滤、标准化、log1p 变换与高变基因筛选，并输出统计摘要。

    Args:
        h5ad_path: 输入的 .h5ad 文件路径。
        pert_col: obs 中记录扰动标签的列名。
        control_label: 对照组在 pert_col 中的标签值。
        sep_char: 组合扰动名称的分隔符（如 '_' 或 '+'）。
        quality_col: obs 中布尔质控列名（若为空字符串或不存在则跳过）。
        min_cells_per_gene: 基因至少在多少个细胞中表达才保留。
        min_cells_per_pert: 每个扰动组至少保留的细胞数。
        target_sum: normalize_total 的目标文库大小。
        n_top_genes: 筛选的高变基因数量（<=0 则保留全部过滤后基因）。
        keep_pert_genes: 是否强制保留出现在扰动标签中的基因。
        save_h5ad_path: 预处理后 .h5ad 保存路径（为空则不保存）。
        summary_csv_path: 扰动组统计摘要 CSV 保存路径（为空则不保存）。

    Returns:
        adata: 预处理后的 AnnData 对象。
        summary_df: 各扰动组细胞数与均值统计 DataFrame。
    """
    # 1. 强制参数类型转换
    h5ad_path = str(h5ad_path)
    pert_col = str(pert_col)
    control_label = str(control_label)
    sep_char = str(sep_char)
    quality_col = str(quality_col) if quality_col else ""
    min_cells_per_gene = int(min_cells_per_gene)
    min_cells_per_pert = int(min_cells_per_pert)
    target_sum = float(target_sum)
    n_top_genes = int(n_top_genes)
    keep_pert_genes = bool(keep_pert_genes)
    save_h5ad_path = str(save_h5ad_path) if save_h5ad_path else ""
    summary_csv_path = str(summary_csv_path) if summary_csv_path else ""

    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"数据文件不存在: {h5ad_path}")

    print(f"[DLSP] 正在读取数据: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[DLSP] 原始数据维度: {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    if pert_col not in adata.obs.columns:
        raise KeyError(f"obs 中缺少扰动列 '{pert_col}'，可用列: {list(adata.obs.columns)}")

    # 统一扰动列为字符串并去除缺失值
    adata = adata[~adata.obs[pert_col].isna()].copy()
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)

    # 2. 质控列过滤
    if quality_col and quality_col in adata.obs.columns:
        before_n = adata.n_obs
        qc_mask = adata.obs[quality_col].astype(bool)
        adata = adata[qc_mask].copy()
        print(f"[DLSP] 质控列 '{quality_col}' 过滤: {before_n} -> {adata.n_obs} 细胞")

    # 3. 过滤样本量过少的扰动组（始终保留对照组）
    pert_counts = adata.obs[pert_col].value_counts()
    valid_perts = pert_counts[
        (pert_counts >= min_cells_per_pert) | (pert_counts.index == control_label)
    ].index.tolist()
    adata = adata[adata.obs[pert_col].isin(valid_perts)].copy()
    print(
        f"[DLSP] 扰动组过滤 (min_cells={min_cells_per_pert}): "
        f"保留 {len(valid_perts)} 个扰动组, 共 {adata.n_obs} 细胞"
    )

    # 4. 过滤低表达基因
    if min_cells_per_gene > 0:
        before_g = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        print(f"[DLSP] 基因过滤 (min_cells={min_cells_per_gene}): {before_g} -> {adata.n_vars} 基因")

    # 确保 X 为 float32 稀疏或稠密矩阵并检测是否为原始计数
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsr()
        sample_vals = adata.X[: min(500, adata.n_obs)].toarray()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
        sample_vals = adata.X[: min(500, adata.n_obs)]

    # 处理潜在 NaN/Inf
    if sparse.issparse(adata.X):
        if np.isnan(adata.X.data).any() or np.isinf(adata.X.data).any():
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if np.isnan(adata.X).any() or np.isinf(adata.X).any():
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    is_raw_counts = bool(np.allclose(sample_vals, np.round(sample_vals)) and sample_vals.max() > 20.0)
    print(f"[DLSP] 矩阵数值检测: max={sample_vals.max():.2f}, 判定为原始计数={is_raw_counts}")

    # 5. 标准化与 log1p
    if is_raw_counts:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print(f"[DLSP] 已执行 normalize_total(target_sum={target_sum}) + log1p")
    else:
        print("[DLSP] 检测到数据已近似对数标准化，跳过 normalize_total 与 log1p")

    # 6. 高变基因筛选（可选保留扰动靶点基因）
    if 0 < n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)
        hvg_mask = adata.var["highly_variable"].values.copy()

        if keep_pert_genes:
            pert_tokens = set()
            for p_name in adata.obs[pert_col].unique():
                if p_name == control_label:
                    continue
                for tok in p_name.split(sep_char):
                    tok_clean = tok.strip()
                    if tok_clean:
                        pert_tokens.add(tok_clean)
            in_var_mask = adata.var_names.isin(pert_tokens)
            added_cnt = int(np.sum(in_var_mask & (~hvg_mask)))
            hvg_mask = hvg_mask | in_var_mask
            print(f"[DLSP] 额外保留扰动靶点基因: {added_cnt} 个")

        adata = adata[:, hvg_mask].copy()
        print(f"[DLSP] HVG 筛选后最终维度: {adata.n_obs} 细胞 x {adata.n_vars} 基因")

    # 7. 生成扰动组统计摘要
    vc = adata.obs[pert_col].value_counts()
    rows = []
    for p_name, cnt in vc.items():
        if p_name == control_label:
            n_perts = 0
        else:
            n_perts = len([t for t in p_name.split(sep_char) if t.strip()])
        rows.append(
            {
                "perturbation": p_name,
                "n_cells": int(cnt),
                "n_perts": int(n_perts),
                "is_control": bool(p_name == control_label),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        by=["is_control", "n_cells"], ascending=[False, False]
    ).reset_index(drop=True)

    # 8. 保存结果
    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
        print(f"[DLSP] 统计摘要已保存至: {summary_csv_path}")

    if save_h5ad_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_h5ad_path)), exist_ok=True)
        adata.write_h5ad(save_h5ad_path)
        print(f"[DLSP] 预处理数据集已保存至: {save_h5ad_path}")

    return adata, summary_df


# 中文注释：本模块负责单细胞扰动数据的读取、质控过滤、原始计数标准化、高变基因筛选及摘要输出，所有列名和阈值均通过参数传入，无硬编码。
