# -*- coding: utf-8 -*-
"""
sota_bench_part4_evaluate_and_plot.py
Unified evaluation & publication-grade visualization comparing:
  Train-Mean, Linear-Ridge, Static-MLP, GEARS-SGConv, scVI,
  scGPT-WholeHuman, Geneformer-V2-104M, scFoundation-100M,
  AIVC v1.1 (Exp31), and AIVC v1.1 Ensemble (Exp34)
on the identical 7-Lineage -> Unseen RPE1 (2,016 perturbations) OOD benchmark.
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions

PRED_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
OUT_CSV = r"D:\data\AIVC\test_results\sota_benchmark_k562_to_rpe1.csv"
OUT_PNG = r"D:\data\AIVC\test_results\sota_benchmark_k562_to_rpe1.png"

MODEL_ORDER = [
    ("Train_Mean", "Train-Mean"),
    ("Static_MLP", "Static-MLP"),
    ("scVI", "scVI (VAE)"),
    ("GEARS_SGConv", "GEARS (SGConv)"),
    ("scFoundation_100M", "scFoundation (100M)"),
    ("Geneformer_V2_104M", "Geneformer-V2 (104M)"),
    ("scGPT", "scGPT (WholeHuman)"),
    ("Linear_Ridge", "Linear-Ridge"),
    ("AIVC_v1.1", "AIVC v1.1 (Ours)"),
    ("AIVC_v1.1_Ensemble", "AIVC v1.1 Ensemble (Ours)"),
]


def main():
    bundle = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    Y_ev = bundle["Y_ev"]
    d_rpe1 = {ev_keys[i]: Y_ev[i] for i in range(len(ev_keys))}

    rows = []
    for file_tag, disp_name in MODEL_ORDER:
        fp = os.path.join(PRED_DIR, f"pred_{file_tag}.npy")
        if not os.path.exists(fp):
            print(f"  [SKIP] Missing {fp}")
            continue
        pmat = np.load(fp)
        df_m = evaluate_perturbation_predictions(
            d_rpe1, {ev_keys[i]: pmat[i] for i in range(len(ev_keys))},
            model_name=disp_name, top_k_list=[20]
        )
        row = {
            "model_id": file_tag,
            "model": disp_name,
            "mean_pearson": float(df_m["pearson_delta_all"].mean()),
            "top20_pearson": float(df_m["pearson_delta_top20"].mean()),
            "dir_acc_top20": float(df_m["dir_acc_top20"].mean()),
            "mse": float(df_m["mse_all"].mean()),
            "n_perts": len(df_m),
        }
        rows.append(row)
        print(
            f"  [{disp_name:25s}] MeanR={row['mean_pearson']:.4f} | "
            f"Top20R={row['top20_pearson']:.4f} | DirAcc={row['dir_acc_top20']:.4f} | MSE={row['mse']:.6f}"
        )

    res_df = pd.DataFrame(rows).sort_values("top20_pearson", ascending=True).reset_index(drop=True)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=180)
    colors = ["#e74c3c" if "AIVC" in m else ("#3498db" if any(k in m for k in ["scGPT", "Geneformer", "scFoundation", "GEARS", "scVI"]) else "#95a5a6") for m in res_df["model"]]

    y_pos = np.arange(len(res_df))
    for ax, col, title in [
        (axes[0], "mean_pearson", "Mean Pearson r (All 2,000 HVGs)"),
        (axes[1], "top20_pearson", "Top-20 DEG Pearson r (Key Effectors)"),
        (axes[2], "dir_acc_top20", "Top-20 Sign Direction Accuracy"),
    ]:
        bars = ax.barh(y_pos, res_df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(res_df["model"] if ax == axes[0] else [""] * len(res_df), fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        xmin = max(0.0, float(res_df[col].min()) - 0.05)
        xmax = float(res_df[col].max()) + 0.06
        ax.set_xlim(xmin, xmax)
        for b, val in zip(bars, res_df[col]):
            ax.text(val + 0.004, b.get_y() + b.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

    fig.suptitle(
        "Head-to-Head SOTA Open-Source Benchmark: 7-Lineage -> Blind Unseen RPE1 (2,016 Perturbations)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved unified SOTA benchmark -> {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
