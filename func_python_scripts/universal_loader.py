# -*- coding: utf-8 -*-
"""
通用多格式上游数据加载器 (Universal Multi-Format Loader)
支持自动识别并加载 .h5ad / .csv / .csv.gz / .tsv / .xlsx / .gct.gz，
对齐目标基因空间，并实现“有突变则加载、无突变则弹性跳过”的多模态兼容接口。
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc


def load_expression_any_format(
    expr_path: str,
    target_genes: list,
    sample_id_col: str = None,
    apply_log1p: bool = True,
    apply_zscore: bool = True,
) -> tuple:
    """加载任意常见格式的单细胞或 Bulk 转录组基态矩阵，对齐到 target_genes。"""
    ext = expr_path.lower()
    if ext.endswith(".h5ad"):
        adata = sc.read_h5ad(expr_path)
        x_max = float(adata.X[:500].max()) if adata.n_obs > 500 else float(adata.X.max())
        if apply_log1p and x_max > 25.0:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        src_idx = {g: i for i, g in enumerate(adata.var_names)}
        common = [(ti, src_idx[g]) for ti, g in enumerate(target_genes) if g in src_idx]
        t_cols = np.array([c[0] for c in common], dtype=np.int64)
        s_cols = np.array([c[1] for c in common], dtype=np.int64)
        X_raw = adata.X[:, s_cols].toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X[:, s_cols])
        mat = np.zeros((adata.n_obs, len(target_genes)), dtype=np.float32)
        mat[:, t_cols] = X_raw.astype(np.float32)
        sample_ids = list(adata.obs_names)
    else:
        if ext.endswith(".gct.gz") or ext.endswith(".gct"):
            df = pd.read_csv(expr_path, sep="\t", skiprows=2)
            gene_col = "Description" if "Description" in df.columns else df.columns[0]
            df = df.drop_duplicates(gene_col).set_index(gene_col)
            df = df.select_dtypes(include=[np.number])
            aligned = df.reindex(target_genes).fillna(0.0).astype(np.float32)
            mat = aligned.values.T
            sample_ids = list(aligned.columns)
        elif ext.endswith(".parquet"):
            df = pd.read_parquet(expr_path)
            sample_ids = df.index.astype(str).tolist()
            aligned = df.reindex(columns=target_genes).fillna(0.0).astype(np.float32)
            mat = aligned.values
        elif ext.endswith(".xlsx") or ext.endswith(".xls"):
            df = pd.read_excel(expr_path)
            id_c = sample_id_col or df.columns[0]
            sample_ids = df[id_c].astype(str).tolist()
            col_map = {str(c).split(" (")[0].strip(): c for c in df.columns if c != id_c}
            mat = np.zeros((len(df), len(target_genes)), dtype=np.float32)
            for gi, g in enumerate(target_genes):
                if g in col_map:
                    mat[:, gi] = df[col_map[g]].fillna(0.0).values.astype(np.float32)
        else:
            sep = "\t" if (".tsv" in ext or ".txt" in ext) else ","
            df = pd.read_csv(expr_path, sep=sep, low_memory=False)
            id_c = sample_id_col or ("ModelID" if "ModelID" in df.columns else df.columns[0])
            sample_ids = df[id_c].astype(str).tolist()
            col_map = {str(c).split(" (")[0].strip(): c for c in df.columns if c != id_c}
            mat = np.zeros((len(df), len(target_genes)), dtype=np.float32)
            for gi, g in enumerate(target_genes):
                if g in col_map:
                    mat[:, gi] = pd.to_numeric(df[col_map[g]], errors="coerce").fillna(0.0).values.astype(np.float32)

        if apply_log1p and float(np.max(mat)) > 25.0:
            mat = np.log1p(np.clip(mat, 0.0, None))

    if apply_zscore:
        mu = mat.mean(axis=1, keepdims=True)
        sd = np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
        mat = (mat - mu) / sd

    return {sid: mat[i] for i, sid in enumerate(sample_ids)}, mat


def load_optional_driver_mutations(
    mut_path: str,
    driver_gene_cols: list,
    sample_id_col: str = "ModelID",
) -> dict:
    """弹性加载可选的体细胞驱动突变文件；若路径为空或不存在，返回空字典以触发纯转录组模式。"""
    if not mut_path or not os.path.exists(mut_path):
        print("[Loader] 未提供突变文件，自动切换至纯转录组弹性预测模式。")
        return {}
    df = pd.read_csv(mut_path) if not mut_path.lower().endswith(".xlsx") else pd.read_excel(mut_path)
    id_c = sample_id_col if sample_id_col in df.columns else df.columns[0]
    col_map = {str(c).split(" (")[0].strip(): c for c in df.columns}
    mat = np.zeros((len(df), len(driver_gene_cols)), dtype=np.float32)
    for gi, g in enumerate(driver_gene_cols):
        g_clean = str(g).split(" (")[0].strip()
        if g in df.columns:
            mat[:, gi] = (pd.to_numeric(df[g], errors="coerce").fillna(0.0).values > 0).astype(np.float32)
        elif g_clean in col_map:
            mat[:, gi] = (pd.to_numeric(df[col_map[g_clean]], errors="coerce").fillna(0.0).values > 0).astype(np.float32)
    return {str(sid): mat[i] for i, sid in enumerate(df[id_c])}
