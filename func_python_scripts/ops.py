# -*- coding: utf-8 -*-
"""
严格防泄漏 OOD 扰动数据划分模块 (OOD Perturbation Split)
支持未见组合扰动 (unseen_combo)、未见单扰动 (unseen_pert) 与跨语境 (unseen_context) 划分。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc


def split_ood_perturbations(
    adata_or_path,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    split_mode: str = "unseen_combo",
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    context_col: str = "",
    val_contexts: str = "",
    test_contexts: str = "",
    seed: int = 42,
    split_col: str = "ood_split",
    subgroup_col: str = "ood_subgroup",
    save_h5ad_path: str = "",
    split_summary_csv: str = "",
):
    """
    对单细胞扰动数据集执行严格防泄漏的 OOD 划分。

    Args:
        adata_or_path: AnnData 对象或 .h5ad 文件路径。
        pert_col: 扰动列名。
        control_label: 对照组标签。
        sep_char: 组合扰动分隔符。
        split_mode: 划分模式 ('unseen_combo', 'unseen_pert', 'unseen_context')。
        val_ratio: 验证集占比。
        test_ratio: 测试集占比。
        context_col: 跨语境划分时的语境列名（如 cell_line）。
        val_contexts: 验证语境（分号分隔字符串）。
        test_contexts: 测试语境（分号分隔字符串）。
        seed: 随机种子。
        split_col: 输出划分列名。
        subgroup_col: 输出精细分组列名。
        save_h5ad_path: 带划分标签的 .h5ad 保存路径。
        split_summary_csv: 划分统计摘要 CSV 路径。

    Returns:
        adata: 更新了 split_col 和 subgroup_col 的 AnnData。
        split_df: 划分统计摘要 DataFrame。
    """
    # 1. 强制类型转换
    pert_col = str(pert_col)
    control_label = str(control_label)
    sep_char = str(sep_char)
    split_mode = str(split_mode).strip().lower()
    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    context_col = str(context_col) if context_col else ""
    val_contexts = str(val_contexts) if val_contexts else ""
    test_contexts = str(test_contexts) if test_contexts else ""
    seed = int(seed)
    split_col = str(split_col)
    subgroup_col = str(subgroup_col)
    save_h5ad_path = str(save_h5ad_path) if save_h5ad_path else ""
    split_summary_csv = str(split_summary_csv) if split_summary_csv else ""

    if isinstance(adata_or_path, str):
        adata = sc.read_h5ad(adata_or_path)
    else:
        adata = adata_or_path.copy()

    rng = np.random.default_rng(seed)
    perts_series = adata.obs[pert_col].astype(str)
    unique_perts = sorted([p for p in perts_series.unique() if p != control_label])

    pert_to_split = {}
    pert_to_subgroup = {}

    if split_mode == "unseen_combo":
        singles = []
        combos = []
        for p in unique_perts:
            tokens = [t.strip() for t in p.split(sep_char) if t.strip()]
            if len(tokens) >= 2:
                combos.append(p)
            else:
                singles.append(p)

        if len(combos) == 0:
            raise ValueError("split_mode='unseen_combo' 但数据中未检测到组合扰动，请检查 sep_char 或使用 'unseen_pert'。")

        # 所有单基因扰动留在训练集，保证模型学习到单基因基元效应
        for p in singles:
            pert_to_split[p] = "train"
            pert_to_subgroup[p] = "train_single"

        combos_arr = np.array(combos)
        perm = rng.permutation(len(combos_arr))
        combos_shuffled = combos_arr[perm]

        n_test = max(1, int(round(len(combos_shuffled) * test_ratio)))
        n_val = max(1, int(round(len(combos_shuffled) * val_ratio)))
        test_combos = set(combos_shuffled[:n_test])
        val_combos = set(combos_shuffled[n_test : n_test + n_val])
        train_combos = set(combos_shuffled[n_test + n_val :])

        # 统计训练集见过的单基因集合
        seen_genes = set()
        for p in singles:
            seen_genes.add(p.strip())
        for p in train_combos:
            for t in p.split(sep_char):
                if t.strip():
                    seen_genes.add(t.strip())

        for p in train_combos:
            pert_to_split[p] = "train"
            pert_to_subgroup[p] = "train_combo"

        for split_name, cset in [("val", val_combos), ("test", test_combos)]:
            for p in cset:
                tokens = [t.strip() for t in p.split(sep_char) if t.strip()]
                n_seen = sum(1 for t in tokens if t in seen_genes)
                pert_to_split[p] = split_name
                pert_to_subgroup[p] = f"{split_name}_combo_seen{n_seen}"

    elif split_mode == "unseen_pert":
        perts_arr = np.array(unique_perts)
        perm = rng.permutation(len(perts_arr))
        perts_shuffled = perts_arr[perm]

        n_test = max(1, int(round(len(perts_shuffled) * test_ratio)))
        n_val = max(1, int(round(len(perts_shuffled) * val_ratio)))
        test_perts = set(perts_shuffled[:n_test])
        val_perts = set(perts_shuffled[n_test : n_test + n_val])
        train_perts = set(perts_shuffled[n_test + n_val :])

        for p in train_perts:
            pert_to_split[p] = "train"
            pert_to_subgroup[p] = "train_unseen_pert"
        for p in val_perts:
            pert_to_split[p] = "val"
            pert_to_subgroup[p] = "val_unseen_pert"
        for p in test_perts:
            pert_to_split[p] = "test"
            pert_to_subgroup[p] = "test_unseen_pert"

    elif split_mode == "unseen_context":
        if not context_col or context_col not in adata.obs.columns:
            raise KeyError(f"unseen_context 模式需要有效的 context_col，当前传入: '{context_col}'")
        val_set = set(x.strip() for x in val_contexts.split(";") if x.strip())
        test_set = set(x.strip() for x in test_contexts.split(";") if x.strip())
        ctx_series = adata.obs[context_col].astype(str)
        split_arr = np.where(
            ctx_series.isin(test_set),
            "test",
            np.where(ctx_series.isin(val_set), "val", "train"),
        )
        adata.obs[split_col] = split_arr
        adata.obs[subgroup_col] = [
            f"{s}_{'ctrl' if p == control_label else 'pert'}"
            for s, p in zip(split_arr, perts_series)
        ]
    else:
        raise ValueError(f"不支持的 split_mode: {split_mode}")

    # 对 unseen_combo / unseen_pert 进行细胞级赋值（对照组按比例分层切分）
    if split_mode in ("unseen_combo", "unseen_pert"):
        split_labels = np.empty(adata.n_obs, dtype=object)
        subgroup_labels = np.empty(adata.n_obs, dtype=object)

        ctrl_idx = np.where(perts_series.values == control_label)[0]
        ctrl_perm = rng.permutation(ctrl_idx)
        n_ctrl_test = max(1, int(round(len(ctrl_perm) * test_ratio)))
        n_ctrl_val = max(1, int(round(len(ctrl_perm) * val_ratio)))

        ctrl_test_idx = ctrl_perm[:n_ctrl_test]
        ctrl_val_idx = ctrl_perm[n_ctrl_test : n_ctrl_test + n_ctrl_val]
        ctrl_train_idx = ctrl_perm[n_ctrl_test + n_ctrl_val :]

        split_labels[ctrl_train_idx] = "train"
        subgroup_labels[ctrl_train_idx] = "train_ctrl"
        split_labels[ctrl_val_idx] = "val"
        subgroup_labels[ctrl_val_idx] = "val_ctrl"
        split_labels[ctrl_test_idx] = "test"
        subgroup_labels[ctrl_test_idx] = "test_ctrl"

        non_ctrl_mask = perts_series.values != control_label
        non_ctrl_perts = perts_series.values[non_ctrl_mask]
        split_labels[non_ctrl_mask] = [pert_to_split[p] for p in non_ctrl_perts]
        subgroup_labels[non_ctrl_mask] = [pert_to_subgroup[p] for p in non_ctrl_perts]

        adata.obs[split_col] = split_labels
        adata.obs[subgroup_col] = subgroup_labels

        # 严格防泄漏校验：非对照扰动标签在 train/val/test 之间交集必须为空
        train_p = set(adata.obs.loc[(adata.obs[split_col] == "train") & (perts_series != control_label), pert_col])
        val_p = set(adata.obs.loc[(adata.obs[split_col] == "val") & (perts_series != control_label), pert_col])
        test_p = set(adata.obs.loc[(adata.obs[split_col] == "test") & (perts_series != control_label), pert_col])
        assert len(train_p & test_p) == 0, "检测到 Train 与 Test 存在扰动泄漏！"
        assert len(train_p & val_p) == 0, "检测到 Train 与 Val 存在扰动泄漏！"
        assert len(val_p & test_p) == 0, "检测到 Val 与 Test 存在扰动泄漏！"
        print("[OPS] 严格防泄漏校验通过: Train / Val / Test 非对照扰动组完全互斥。")

    # 生成统计摘要
    summary_rows = []
    for (sp, sg), grp in adata.obs.groupby([split_col, subgroup_col]):
        n_unique_perts = grp.loc[grp[pert_col] != control_label, pert_col].nunique()
        summary_rows.append(
            {
                "split": sp,
                "subgroup": sg,
                "n_cells": int(len(grp)),
                "n_unique_non_ctrl_perts": int(n_unique_perts),
            }
        )
    split_df = pd.DataFrame(summary_rows).sort_values(["split", "subgroup"]).reset_index(drop=True)
    print("[OPS] 划分统计:\n", split_df.to_string(index=False))

    if split_summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(split_summary_csv)), exist_ok=True)
        split_df.to_csv(split_summary_csv, index=False, encoding="utf-8")
        print(f"[OPS] 划分统计表已保存至: {split_summary_csv}")

    if save_h5ad_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_h5ad_path)), exist_ok=True)
        adata.write_h5ad(save_h5ad_path)
        print(f"[OPS] 带划分标签的 AnnData 已保存至: {save_h5ad_path}")

    return adata, split_df


# 中文注释：本模块实现单细胞扰动数据的严格防泄漏 OOD 划分，支持未见组合、未见单扰动及跨语境三种基准划分模式。
