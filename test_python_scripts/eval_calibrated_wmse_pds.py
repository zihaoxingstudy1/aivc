# -*- coding: utf-8 -*-
"""
eval_calibrated_wmse_pds.py
Compute Delta-Gap Calibrated Perturbation Metrics (WMSE, Perturbation Discrimination Score PDS)
across all 10 models + No-Change baseline on blind unseen RPE1 (n = 2,016 perturbations).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRED_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
OUT_CSV = r"D:\data\AIVC\test_results\sota_benchmark_rpe1_calibrated_wmse_pds.csv"


def compute_wmse_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Weighted MSE where each gene is weighted by |y_true| so zero-delta genes cannot inflate score."""
    abs_t = np.abs(y_true)
    w = abs_t / (abs_t.sum(axis=1, keepdims=True) + 1e-8)
    return np.sum(w * (y_true - y_pred) ** 2, axis=1)


def compute_pds_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Perturbation Discrimination Score: 1 - rank(cos_dist(pred_i, true_i)) / (N - 1)."""
    nt = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-8)
    np_ = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
    # Cosine similarity matrix (N x N): row i = pred_i, col j = true_j
    sim = np_ @ nt.T
    dist = 1.0 - sim
    n = y_true.shape[0]
    diag = np.diag(dist)[:, None]
    # Mid-rank tie breaking so constant / zero predictions get exact random 0.500
    ranks = np.sum(dist < diag, axis=1) + 0.5 * np.maximum(np.sum(dist == diag, axis=1) - 1, 0)
    return 1.0 - ranks / float(n - 1)


def main():
    bundle = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"))
    y_true = bundle["Y_ev"].astype(np.float32)
    n = y_true.shape[0]
    print(f"Loaded RPE1 blind test ground truth: {y_true.shape}")

    models = {
        "No-Change": np.zeros_like(y_true),
        "Train-Mean": np.load(os.path.join(PRED_DIR, "pred_Train_Mean.npy")),
        "Static-MLP": np.load(os.path.join(PRED_DIR, "pred_Static_MLP.npy")),
        "Linear-Ridge": np.load(os.path.join(PRED_DIR, "pred_Linear_Ridge.npy")),
        "scVI": np.load(os.path.join(PRED_DIR, "pred_scVI.npy")),
        "GEARS-SGConv": np.load(os.path.join(PRED_DIR, "pred_GEARS_SGConv.npy")),
        "Geneformer-V2-104M": np.load(os.path.join(PRED_DIR, "pred_Geneformer_V2_104M.npy")),
        "scGPT": np.load(os.path.join(PRED_DIR, "pred_scGPT.npy")),
        "scFoundation-100M": np.load(os.path.join(PRED_DIR, "pred_scFoundation_100M.npy")),
        "AIVC v1.1": np.load(os.path.join(PRED_DIR, "pred_AIVC_v1.1.npy")),
        "AIVC v1.1 Ensemble": np.load(os.path.join(PRED_DIR, "pred_AIVC_v1.1_Ensemble.npy")),
    }

    rows = []
    wmse_map, pds_map = {}, {}
    for name, yp in models.items():
        yp = yp.astype(np.float32)
        wmse_i = compute_wmse_per_pert(y_true, yp)
        pds_i = compute_pds_per_pert(y_true, yp)
        mse_i = np.mean((y_true - yp) ** 2, axis=1)
        wmse_map[name] = wmse_i
        pds_map[name] = pds_i
        rows.append({
            "Model": name,
            "WMSE_mean": float(np.mean(wmse_i)),
            "WMSE_median": float(np.median(wmse_i)),
            "PDS_mean": float(np.mean(pds_i)),
            "PDS_median": float(np.median(pds_i)),
            "Top1_Match_Pct": float(np.mean((pds_i == 1.0) & (np.linalg.norm(yp, axis=1) > 1e-8)) * 100.0),
            "MSE_global": float(np.mean(mse_i)),
        })

    df = pd.DataFrame(rows)
    # Compute Wilcoxon p-values of AIVC v1.1 Ensemble vs each baseline
    ens_wmse = wmse_map["AIVC v1.1 Ensemble"]
    ens_pds = pds_map["AIVC v1.1 Ensemble"]
    p_w_list, p_p_list = [], []
    for name in df["Model"]:
        if name == "AIVC v1.1 Ensemble":
            p_w_list.append(1.0)
            p_p_list.append(1.0)
        else:
            _, pw = wilcoxon(ens_wmse, wmse_map[name], alternative="less")
            _, pp = wilcoxon(ens_pds, pds_map[name], alternative="greater")
            p_w_list.append(float(pw))
            p_p_list.append(float(pp))
    df["Wilcoxon_p_WMSE_vs_Ens"] = p_w_list
    df["Wilcoxon_p_PDS_vs_Ens"] = p_p_list
    df = df.sort_values("WMSE_mean", ascending=True).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Calibrated Delta-Gap Benchmark (Blind RPE1, n=2,016) ===")
    print(df[["Model", "WMSE_mean", "PDS_mean", "PDS_median", "Top1_Match_Pct", "MSE_global", "Wilcoxon_p_WMSE_vs_Ens"]].to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
