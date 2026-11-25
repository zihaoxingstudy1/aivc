# -*- coding: utf-8 -*-
"""
多细胞系联合数据准备与攻克 Replogle (K562 -> RPE1) 跨细胞系泛化模块
将 Replogle K562、Norman K562、Shifrut Primary T、Papalexi THP-1 四个数据集对齐合并，
以完全未见的 Replogle RPE1 作为跨细胞系 Zero-shot 测试集。
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse


def _clean_and_std(adata, ctx_name: str, pert_col: str, control_label: str, strip_guide_suffix: bool = False):
    """清洗单个数据集为统一格式并执行原始计数标准化。"""
    a = adata.copy()
    a.var_names_make_unique()
    a.obs_names_make_unique()
    a.obs["cell_line"] = str(ctx_name)
    perts = a.obs[pert_col].astype(str)
    if strip_guide_suffix:
        perts = perts.apply(lambda s: s if s == control_label else re.sub(r"g\d+$", "", s))
    a.obs["perturbation"] = perts
    a.obs = a.obs[["cell_line", "perturbation"]].copy()

    if sparse.issparse(a.X):
        a.X = a.X.astype(np.float32).tocsr()
        s_max = float(a.X[: min(200, a.n_obs)].toarray().max())
    else:
        a.X = sparse.csr_matrix(np.asarray(a.X, dtype=np.float32))
        s_max = float(a.X[: min(200, a.n_obs)].toarray().max())

    if s_max > 20.0:
        sc.pp.normalize_total(a, target_sum=10000.0)
        sc.pp.log1p(a)
    return a


def build_multi_lineage_replogle_dataset(
    replogle_split_h5ad: str,
    norman_h5ad: str,
    shifrut_h5ad: str,
    papalexi_h5ad: str,
    save_h5ad_path: str,
):
    """
    构建多谱系联合训练集 + 纯未见 RPE1 测试集的跨细胞系基准数据集。
    严格对齐到 Replogle_K562_RPE1_hvg2000_split.h5ad 的基因空间，保证可比性。
    """
    print("[MCJT] 正在读取 Replogle 基准数据集...")
    rep = sc.read_h5ad(replogle_split_h5ad)
    target_genes = list(rep.var_names)
    target_gene_set = set(target_genes)

    # 提取 Replogle 中的 K562 (train) 和 RPE1 (test)
    rep_k562 = rep[rep.obs["cell_line"] == "K562"].copy()
    rep_k562.obs = rep_k562.obs[["cell_line", "perturbation", "ood_split", "ood_subgroup"]].copy()
    rep_rpe1 = rep[rep.obs["cell_line"] == "RPE1"].copy()
    rep_rpe1.obs = rep_rpe1.obs[["cell_line", "perturbation", "ood_split", "ood_subgroup"]].copy()

    # 读取并标准化其他三个数据集
    print("[MCJT] 正在读取并标准化 Norman(K562), Shifrut(Primary_T), Papalexi(THP-1)...")
    norman = _clean_and_std(sc.read_h5ad(norman_h5ad), "Norman_K562", "perturbation", "control", False)
    shifrut = _clean_and_std(sc.read_h5ad(shifrut_h5ad), "Primary_T", "perturbation", "control", False)
    papalexi = _clean_and_std(sc.read_h5ad(papalexi_h5ad), "THP1", "perturbation", "control", True)

    extra_list = []
    for extra in [norman, shifrut, papalexi]:
        # 过滤掉样本过少的扰动组
        vc = extra.obs["perturbation"].value_counts()
        keep_p = set(vc[(vc >= 30) | (vc.index == "control")].index)
        extra = extra[extra.obs["perturbation"].isin(keep_p)].copy()
        extra.obs["ood_split"] = "train"
        extra.obs["ood_subgroup"] = np.where(extra.obs["perturbation"] == "control", "train_ctrl", "train_extra_lineage")
        extra_list.append(extra)

    # 找所有数据集的共有基因（以 Replogle HVG 为主干，确保所有训练/测试基因一致）
    common_genes = target_gene_set.copy()
    for ex in extra_list:
        common_genes &= set(ex.var_names)
    common_genes = [g for g in target_genes if g in common_genes]
    print(f"[MCJT] 跨 4 数据集对齐共有基因数: {len(common_genes)} / {len(target_genes)}")

    parts = [rep_k562[:, common_genes].copy(), rep_rpe1[:, common_genes].copy()]
    for ex in extra_list:
        parts.append(ex[:, common_genes].copy())

    merged = ad.concat(parts, join="inner", index_unique="-")
    # 保留原始 var 里的 ensembl_id 信息
    if "ensembl_id" in rep.var.columns:
        merged.var["ensembl_id"] = rep.var.loc[common_genes, "ensembl_id"].values

    print("[MCJT] 合并后各语境统计:")
    print(merged.obs.groupby(["ood_split", "cell_line"]).size())

    os.makedirs(os.path.dirname(os.path.abspath(save_h5ad_path)), exist_ok=True)
    merged.write_h5ad(save_h5ad_path)
    print(f"[MCJT] 多细胞系联合数据集已保存至: {save_h5ad_path}")
    return merged


# 中文注释：本模块将 Replogle K562、Norman K562、Shifrut 原代 T 细胞与 Papalexi THP-1 单核细胞对齐合并，构建多谱系联合训练集以攻克 RPE1 跨细胞系泛化。
