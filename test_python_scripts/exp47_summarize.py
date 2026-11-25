# -*- coding: utf-8 -*-
"""Summarise exp47 arms: ensemble metrics, weak/moderate/strong tertiles, paired deltas."""
import os
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "test_results", "sota_bench_preds")
EXP = os.path.join(ROOT, "test_results")


def load_arm(arm):
    fs = sorted(glob.glob(os.path.join(EXP, f"exp47_pred_{arm}_s*.npy")))
    return [np.load(f) for f in fs]


def _r(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(Y, P):
    n = Y.shape[0]
    mr = np.array([_r(Y[i], P[i]) for i in range(n)])
    t20 = np.empty(n); da = np.empty(n)
    for i in range(n):
        idx = np.argsort(np.abs(Y[i]))[-20:]
        t20[i] = _r(Y[i][idx], P[i][idx])
        da[i] = np.mean(np.sign(Y[i][idx]) == np.sign(P[i][idx]))
    return mr, t20, da


def main():
    b = np.load(os.path.join(PRED, "rpe1_ood_bundle.npz"), allow_pickle=True)
    Y = b["Y_ev"]
    order = np.argsort(np.linalg.norm(Y, axis=1))
    q = np.array_split(order, 3)
    tert = {"weak": q[0], "moderate": q[1], "strong": q[2]}

    arms = ["signed", "unsigned_abs", "unsigned_sum", "shuffled", "none",
            "causal_only", "neither"]
    rows = []
    per_seed = {}
    ens = {}
    for a in arms:
        ps = load_arm(a)
        if not ps:
            continue
        E = np.mean(ps, axis=0)
        ens[a] = E
        mr, t20, da = metrics(Y, E)
        row = {"arm": a, "seeds": len(ps), "ensemble_mean_r": mr.mean(), "ensemble_top20": t20.mean(),
               "ensemble_dir": da.mean()}
        for k, idx in tert.items():
            row[f"top20_{k}"] = t20[idx].mean()
        rows.append(row)
        per_seed[a] = [metrics(Y, p)[1].mean() for p in ps]

    df = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 200)
    print("\n=== ensemble per arm ===")
    print(df.round(4).to_string())

    print("\n=== per-seed Top-20 (n=5) ===")
    print(pd.DataFrame(per_seed).round(4).to_string())

    print("\n=== paired deltas vs signed (per seed) ===")
    base = np.array(per_seed["signed"])
    for a in arms:
        if a == "signed" or a not in per_seed:
            continue
        d = base - np.array(per_seed[a])
        print(f"{a:13s} meanΔ={d.mean():+.4f}  sd={d.std(ddof=1):.4f}  "
              f"seeds_worse={int((d > 0).sum())}/5  per-seed={np.round(d, 4).tolist()}")

    print("\n=== prediction correlation between arms (ensemble) ===")
    keys = list(ens)
    M = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for i in keys:
        for j in keys:
            M.loc[i, j] = float(np.corrcoef(ens[i].ravel(), ens[j].ravel())[0, 1])
    print(M.round(4).to_string())

    print("\n=== causal block magnitude (mean |0.5*prog| relative to |ridge|) ===")
    # quick probe: spread of ensemble predictions across arms tells whether the block is used
    for a in keys:
        print(f"{a:13s} mean|pred|={np.abs(ens[a]).mean():.6f}")


if __name__ == "__main__":
    main()