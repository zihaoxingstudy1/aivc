# -*- coding: utf-8 -*-
"""
线性基线与多维评测基准模块 (Linear Baseline & Benchmark)
实现 Control-Mean、Train-Mean、Linear-Ridge 三大基线，并计算全基因与 Top-DEG 扰动评测指标。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _to_dense_array(mat):
    """将稀疏或稠密矩阵安全转为 float32 numpy 数组。"""
    if sparse.issparse(mat):
        return mat.toarray().astype(np.float32)
    return np.asarray(mat, dtype=np.float32)


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算一维向量的安全 Pearson 相关系数（方差为 0 时返回 0.0）。"""
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if np.isfinite(r) else 0.0


def _compute_group_deltas(adata_sub, pert_col: str, mu_ctrl: np.ndarray) -> dict:
    """使用稀疏指示矩阵高效计算各扰动组的均值偏移 Delta。"""
    perts = adata_sub.obs[pert_col].astype(str).values
    unique_perts = sorted(np.unique(perts))
    pert_to_id = {p: i for i, p in enumerate(unique_perts)}
    row_ids = np.array([pert_to_id[p] for p in perts], dtype=np.int32)
    col_ids = np.arange(len(perts), dtype=np.int32)
    counts = np.bincount(row_ids, minlength=len(unique_perts)).astype(np.float32)
    weights = 1.0 / np.maximum(counts[row_ids], 1.0)
    ind_mat = sparse.csr_matrix((weights, (row_ids, col_ids)), shape=(len(unique_perts), len(perts)), dtype=np.float32)
    group_means = _to_dense_array(ind_mat @ adata_sub.X)
    return {p: group_means[i] - mu_ctrl for i, p in enumerate(unique_perts)}


def evaluate_perturbation_predictions(
    true_deltas: dict,
    pred_deltas: dict,
    model_name: str,
    top_k_list: tuple = (20, 50),
):
    """
    评估单个模型在所有测试扰动上的预测表现。

    Args:
        true_deltas: {pert_name: 1D numpy array of true delta}
        pred_deltas: {pert_name: 1D numpy array of predicted delta}
        model_name: 模型名称。
        top_k_list: 评估 Top-K DEG 的 K 值列表。

    Returns:
        per_pert_df: 每个扰动的详细指标 DataFrame。
    """
    rows = []
    for pert, dt in true_deltas.items():
        dp = pred_deltas[pert]
        mse_all = float(np.mean((dt - dp) ** 2))
        r_all = _safe_pearson(dt, dp)

        row = {
            "model": str(model_name),
            "perturbation": str(pert),
            "mse_all": mse_all,
            "pearson_delta_all": r_all,
        }

        abs_dt = np.abs(dt)
        for k in top_k_list:
            k_eff = min(int(k), len(dt))
            top_idx = np.argsort(abs_dt)[-k_eff:]
            dt_top = dt[top_idx]
            dp_top = dp[top_idx]
            row[f"pearson_delta_top{k_eff}"] = _safe_pearson(dt_top, dp_top)
            row[f"mse_top{k_eff}"] = float(np.mean((dt_top - dp_top) ** 2))
            row[f"dir_acc_top{k_eff}"] = float(np.mean(np.sign(dt_top) == np.sign(dp_top)))

        rows.append(row)

    return pd.DataFrame(rows)


def run_linear_baseline_benchmark(
    adata_or_path,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    split_col: str = "ood_split",
    train_split: str = "train",
    eval_split: str = "test",
    ridge_alpha: float = 10.0,
    top_k_str: str = "20;50",
    per_pert_csv_path: str = "",
    summary_csv_path: str = "",
    plot_png_path: str = "",
):
    """
    在已划分的扰动数据集上运行三大基线（Control-Mean, Train-Mean, Linear-Ridge）并输出基准报告。

    Args:
        adata_or_path: 带划分标签的 AnnData 对象或 .h5ad 路径。
        pert_col: 扰动列名。
        control_label: 对照组标签。
        sep_char: 组合扰动分隔符。
        split_col: 划分列名。
        train_split: 训练集标签。
        eval_split: 评测集标签（通常为 'val' 或 'test'）。
        ridge_alpha: Ridge 回归正则化系数。
        top_k_str: 分号分隔的 Top-K DEG 列表（如 '20;50'）。
        per_pert_csv_path: 各扰动详细指标保存路径。
        summary_csv_path: 模型汇总对比表保存路径。
        plot_png_path: 可视化对比图保存路径。

    Returns:
        summary_df: 各基线汇总对比 DataFrame。
        per_pert_df: 各扰动详细得分 DataFrame。
    """
    # 1. 强制类型转换
    pert_col = str(pert_col)
    control_label = str(control_label)
    sep_char = str(sep_char)
    split_col = str(split_col)
    train_split = str(train_split)
    eval_split = str(eval_split)
    ridge_alpha = float(ridge_alpha)
    top_k_list = tuple(int(x.strip()) for x in str(top_k_str).split(";") if x.strip())

    if isinstance(adata_or_path, str):
        adata = sc.read_h5ad(adata_or_path)
    else:
        adata = adata_or_path

    if split_col not in adata.obs.columns:
        raise KeyError(f"obs 中缺少划分列 '{split_col}'")

    train_adata = adata[adata.obs[split_col] == train_split]
    eval_adata = adata[adata.obs[split_col] == eval_split]

    # 2. 计算训练集与评测集各自 Control 均值
    train_ctrl_mask = (train_adata.obs[pert_col].astype(str) == control_label).values
    eval_ctrl_mask = (eval_adata.obs[pert_col].astype(str) == control_label).values
    if train_ctrl_mask.sum() == 0 or eval_ctrl_mask.sum() == 0:
        raise ValueError("训练集或评测集中缺少对照组细胞！")

    mu_ctrl_train = _to_dense_array(train_adata.X[train_ctrl_mask]).mean(axis=0)
    mu_ctrl_eval = _to_dense_array(eval_adata.X[eval_ctrl_mask]).mean(axis=0)

    # 3. 计算训练集各扰动组的真实 Delta（用于拟合基线）
    train_non_ctrl = train_adata[~train_ctrl_mask]
    train_pert_deltas = _compute_group_deltas(train_non_ctrl, pert_col, mu_ctrl_train)
    unique_train_perts = sorted(train_pert_deltas.keys())

    # 4. 计算评测集各扰动组的真实 Delta（Ground Truth）
    eval_non_ctrl = eval_adata[~eval_ctrl_mask]
    eval_true_deltas = _compute_group_deltas(eval_non_ctrl, pert_col, mu_ctrl_eval)
    unique_eval_perts = sorted(eval_true_deltas.keys())

    print(
        f"[LBB] 训练集扰动组: {len(unique_train_perts)} 个 | "
        f"评测集 ({eval_split}) 扰动组: {len(unique_eval_perts)} 个"
    )

    # 5. 构建 Multi-Hot 扰动特征编码器（基于训练集见过的单基因词表）
    vocab_tokens = sorted(
        {
            tok.strip()
            for p in unique_train_perts
            for tok in p.split(sep_char)
            if tok.strip()
        }
    )
    token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}

    def encode_perts(pert_list):
        mat = np.zeros((len(pert_list), len(vocab_tokens)), dtype=np.float32)
        for row_i, p_name in enumerate(pert_list):
            for tok in p_name.split(sep_char):
                t = tok.strip()
                if t in token_to_idx:
                    mat[row_i, token_to_idx[t]] = 1.0
        return mat

    X_train_enc = encode_perts(unique_train_perts)
    Y_train_delta = np.vstack([train_pert_deltas[p] for p in unique_train_perts])

    X_eval_enc = encode_perts(unique_eval_perts)

    # 基线 1: Control-Mean (Delta = 0)
    pred_ctrl_mean = {p: np.zeros_like(mu_ctrl_eval) for p in unique_eval_perts}

    # 基线 2: Train-Mean (Delta = 训练集所有扰动平均偏移)
    global_mean_delta = Y_train_delta.mean(axis=0)
    pred_train_mean = {p: global_mean_delta.copy() for p in unique_eval_perts}

    # 基线 3: Linear-Ridge (扰动 Multi-Hot 岭回归线性叠加模型)
    ridge_model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge_model.fit(X_train_enc, Y_train_delta)
    Y_eval_ridge = ridge_model.predict(X_eval_enc)
    pred_ridge = {p: Y_eval_ridge[i] for i, p in enumerate(unique_eval_perts)}

    # 6. 评估三大基线
    df_list = [
        evaluate_perturbation_predictions(eval_true_deltas, pred_ctrl_mean, "Control-Mean", top_k_list),
        evaluate_perturbation_predictions(eval_true_deltas, pred_train_mean, "Train-Mean", top_k_list),
        evaluate_perturbation_predictions(eval_true_deltas, pred_ridge, f"Linear-Ridge(a={ridge_alpha:g})", top_k_list),
    ]
    per_pert_df = pd.concat(df_list, ignore_index=True)

    metric_cols = [c for c in per_pert_df.columns if c not in ("model", "perturbation")]
    summary_rows = []
    for model_name, grp in per_pert_df.groupby("model", sort=False):
        srow = {"model": model_name, "n_eval_perts": int(len(grp))}
        for mc in metric_cols:
            srow[f"{mc}_mean"] = float(grp[mc].mean())
            srow[f"{mc}_std"] = float(grp[mc].std())
        summary_rows.append(srow)
    summary_df = pd.DataFrame(summary_rows)

    print("\n[LBB] 基准测试汇总结果:")
    disp_cols = ["model", "pearson_delta_all_mean", "mse_all_mean"]
    for k in top_k_list:
        disp_cols.extend([f"pearson_delta_top{k}_mean", f"dir_acc_top{k}_mean"])
    print(summary_df[disp_cols].to_string(index=False))

    # 7. 保存表格与绘图
    if per_pert_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(per_pert_csv_path)), exist_ok=True)
        per_pert_df.to_csv(per_pert_csv_path, index=False, encoding="utf-8")
        print(f"[LBB] 逐扰动评测表已保存至: {per_pert_csv_path}")

    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
        print(f"[LBB] 汇总对比表已保存至: {summary_csv_path}")

    if plot_png_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_png_path)), exist_ok=True)
        _plot_benchmark_comparison(per_pert_df, top_k_list, eval_split, plot_png_path)
        print(f"[LBB] 基准对比图已保存至: {plot_png_path}")

    return summary_df, per_pert_df


def _plot_benchmark_comparison(per_pert_df: pd.DataFrame, top_k_list: tuple, eval_split: str, plot_png_path: str):
    """绘制三大基线在核心指标上的对比柱状图（含标准误误差棒与数值标签）。"""
    sns.set_style("whitegrid")
    k0 = top_k_list[0] if len(top_k_list) > 0 else 20
    metrics_to_plot = [
        ("pearson_delta_all", "Pearson(Delta) - All Genes (Higher Better)"),
        (f"pearson_delta_top{k0}", f"Pearson(Delta) - Top{k0} DEGs (Higher Better)"),
        ("mse_all", "MSE(Delta) - All Genes (Lower Better)"),
        (f"dir_acc_top{k0}", f"Direction Acc - Top{k0} DEGs (Higher Better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    axes = axes.flatten()
    palette = ["#95a5a6", "#3498db", "#e74c3c"]

    for idx, (col, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if col not in per_pert_df.columns:
            continue
        sns.barplot(
            data=per_pert_df,
            x="model",
            y=col,
            hue="model",
            ax=ax,
            palette=palette[: per_pert_df["model"].nunique()],
            errorbar="se",
            capsize=0.15,
            legend=False,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=10)
        ax.tick_params(axis="x", rotation=10)

        for p in ax.patches:
            h = p.get_height()
            if np.isfinite(h):
                ax.annotate(
                    f"{h:.4f}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 4),
                    textcoords="offset points",
                )

    fig.suptitle(f"AIVC Linear Baseline Benchmark ({eval_split} split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_png_path, bbox_inches="tight")
    plt.close(fig)


# 中文注释：本模块实现 Control-Mean、Train-Mean 和 Linear-Ridge 三大基线模型及全基因/Top-DEG 多维评测基准，并生成结果表格与对比可视化。
