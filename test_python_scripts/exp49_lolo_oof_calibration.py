# -*- coding: utf-8 -*-
"""
exp49_lolo_oof_calibration.py

Purpose
-------
AIVC's test predictions are systematically over-shrunk: prediction SD is only
0.30x the true delta SD on blind RPE1. A global rescaling and a modest blend
with the ridge anchor could recover a large part of WMSE / mean-r without
touching ranking (Top-20 r and PDS are invariant to a positive scalar).

Doing that calibration on the test set would be test-set tuning. Instead we
generate strictly out-of-fold (OOF) predictions on the 6,918 training
perturbations by leave-one-lineage-out (LOLO) retraining, fit the calibration
parameters **on the OOF predictions only**, freeze them, and apply them once to
the published test predictions.

Fitted parameters
-----------------
  s_global : scalar, least squares P*s ~ Y on OOF
  s_gene   : per-gene scalar with shrinkage lambda toward 1.0
  w_blend  : AIVC vs ridge mixing weight, selected on OOF Top-20 r
  combos   : blend -> global scale -> per-gene scale

Arms evaluated once on blind RPE1 (n = 2,016)
---------------------------------------------
  A0 original
  A1 global scale
  A2 blend
  A3 blend + global scale
  A4 per-gene scale (shrinkage)
  A5 blend + global scale + per-gene scale

Run
---
  python test_python_scripts/exp49_lolo_oof_calibration.py
"""

import os
import sys
import time
import types

for _m in ("gseapy", "scanpy"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "func_python_scripts"))
sys.path.insert(0, os.path.join(ROOT, "test_python_scripts"))

# reuse the exact model / feature / prior construction of exp47
import exp47_sign_ablation as E47  # noqa: E402
from eval_calibrated_wmse_pds import (  # noqa: E402
    compute_pds_per_pert, compute_wmse_per_pert,
)

PRED_DIR = os.path.join(ROOT, "test_results", "sota_bench_preds")
OUT_DIR = os.path.join(ROOT, "test_results")
OOF_NPY = os.path.join(OUT_DIR, "exp49_oof_pred_signed.npy")
OUT_CSV = os.path.join(OUT_DIR, "exp49_calibration_results.csv")
PARAM_JSON = os.path.join(OUT_DIR, "exp49_frozen_params.csv")

SEED = 42


def mean_r(Y, P):
    out = np.empty(len(Y))
    for i in range(len(Y)):
        a, b = Y[i], P[i]
        out[i] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else np.corrcoef(a, b)[0, 1]
    return out


def top_r(Y, P, k=20):
    out = np.empty(len(Y))
    for i in range(len(Y)):
        idx = np.argsort(np.abs(Y[i]))[-k:]
        a, b = Y[i][idx], P[i][idx]
        out[i] = 0.0 if (a.std() < 1e-12 or b.std() < 1e-12) else np.corrcoef(a, b)[0, 1]
    return out


def dir_acc(Y, P, k=20):
    out = np.empty(len(Y))
    for i in range(len(Y)):
        idx = np.argsort(np.abs(Y[i]))[-k:]
        out[i] = float(np.mean(np.sign(Y[i][idx]) == np.sign(P[i][idx])))
    return out


def report(name, Y, P):
    mr = mean_r(Y, P).mean()
    tr = top_r(Y, P).mean()
    da = dir_acc(Y, P).mean()
    mse = float(np.mean((Y - P) ** 2))
    w = float(np.mean(compute_wmse_per_pert(Y, P)))
    p = float(np.mean(compute_pds_per_pert(Y, P)))
    return dict(arm=name, mean_r=mr, top20_r=tr, dir_acc=da, mse=mse, wmse=w, pds=p)


def build_oof(b, genes, g2i, ctrls, A_signed, Xl, vidx):
    """Leave-one-lineage-out retraining; returns OOF predictions (6918 x G)."""
    tr_perts = [str(p) for p in b["tr_perts"]]
    tr_lin = np.array([str(x) for x in b["tr_lineages"]])
    Y_tr = b["Y_tr"]
    n_genes = len(genes)

    # global LOLO ridge anchors (same convention as exp47)
    pred_ridge_tr = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lin):
        m_out = tr_lin == ln
        m_in = ~m_out
        pred_ridge_tr[m_out] = Ridge(alpha=50.0).fit(
            Xl[m_in], Y_tr[m_in]).predict(Xl[m_out]).astype(np.float32)

    print("[exp49] inferring baseline protein/phosphosite states ...", flush=True)
    pp = E47.predict_lineage_prot_phos(ctrls, genes)

    oof = np.zeros_like(Y_tr)
    lineages = list(np.unique(tr_lin))
    for fi, ln in enumerate(lineages):
        t0 = time.time()
        m_out = tr_lin == ln
        m_in = ~m_out
        Y_in = Y_tr[m_in]
        # refit the programme basis on the training folds only (no leakage)
        U = TruncatedSVD(n_components=64, random_state=SEED).fit(Y_in).components_.astype(np.float32)
        cmat = E47.build_causal_matrix(A_signed, U)

        f_tr = E47.build_features(
            [p for p, m in zip(tr_perts, m_in) if m],
            [l for l, m in zip(tr_lin, m_in) if m],
            pred_ridge_tr[m_in], b["X_esm_tr"][m_in], b["X_coexp_tr"][m_in],
            ctrls, g2i, pp, cmat, True, True)
        f_ev = E47.build_features(
            [p for p, m in zip(tr_perts, m_out) if m],
            [l for l, m in zip(tr_lin, m_out) if m],
            pred_ridge_tr[m_out], b["X_esm_tr"][m_out], b["X_coexp_tr"][m_out],
            ctrls, g2i, pp, cmat, True, True)

        preds = E47.train_predict(f_tr, f_ev, Y_in, n_genes, U, [SEED])
        oof[m_out] = preds[0]
        print(f"[exp49] fold {fi+1}/{len(lineages)} held out {ln:9s} "
              f"n={m_out.sum():4d} Top20={top_r(Y_tr[m_out], preds[0]).mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return oof, pred_ridge_tr


def fit_gene_scale(P, Y, lam):
    num = (P * Y).sum(axis=0)
    den = (P * P).sum(axis=0)
    return (num + lam) / (den + lam + 1e-12)


def main():
    t0 = time.time()
    b = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    genes = [str(g) for g in b["target_genes"]]
    g2i = {g: i for i, g in enumerate(genes)}
    ctrls = {str(k): v for k, v in zip(b["ctrl_names"], b["ctrl_vecs"])}
    Y_tr = b["Y_tr"].astype(np.float64)
    Y_ev = b["Y_ev"].astype(np.float64)
    tr_perts = [str(p) for p in b["tr_perts"]]

    vocab = sorted(set(tr_perts))
    vidx = {t: i for i, t in enumerate(vocab)}
    Xl = np.zeros((len(tr_perts), len(vocab)), dtype=np.float32)
    for ri, p in enumerate(tr_perts):
        Xl[ri, vidx[p]] = 1.0

    A_pos, A_neg, n_pos, n_neg = E47.build_prior_from_tsv(genes, E47.TSV)
    print(f"[exp49] prior edges {n_pos}/{n_neg} (expect {E47.EXPECT_POS}/{E47.EXPECT_NEG})", flush=True)
    A_signed = (A_pos - A_neg).astype(np.float32)

    if os.path.exists(OOF_NPY):
        oof = np.load(OOF_NPY).astype(np.float64)
        print("[exp49] reusing cached OOF predictions", flush=True)
    else:
        oof, pred_ridge_tr = build_oof(b, genes, g2i, ctrls, A_signed, Xl, vidx)
        np.save(OOF_NPY, oof.astype(np.float32))

    # ridge OOF anchors for blending (recompute to keep alignment)
    tr_lin = np.array([str(x) for x in b["tr_lineages"]])
    ridge_oof = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lin):
        m_out = tr_lin == ln
        m_in = ~m_out
        ridge_oof[m_out] = Ridge(alpha=50.0).fit(Xl[m_in], Y_tr[m_in]).predict(Xl[m_out])

    print("\n=== OOF (6,918 training perturbations, LOLO) ===")
    print(report("OOF AIVC", Y_tr, oof))
    print(report("OOF ridge", Y_tr, ridge_oof))

    # ---------- fit parameters on OOF ----------
    s_global = float((oof * Y_tr).sum() / ((oof * oof).sum() + 1e-12))

    best_w, best_score = 1.0, -9
    blend_rows = []
    for w in np.arange(0.0, 1.0001, 0.05):
        bl = w * oof + (1 - w) * ridge_oof
        tr = top_r(Y_tr, bl).mean()
        blend_rows.append((round(float(w), 2), tr, mean_r(Y_tr, bl).mean()))
        if tr > best_score:
            best_score, best_w = tr, float(w)
    w_blend = best_w

    lam_grid = [0.0, 10.0, 100.0, 1000.0, 10000.0]
    lam_best, lam_score = 0.0, -9
    for lam in lam_grid:
        s = fit_gene_scale(oof, Y_tr, lam)
        tr = top_r(Y_tr, oof * s[None, :]).mean()
        if tr > lam_score:
            lam_score, lam_best = tr, lam
    s_gene_oof = fit_gene_scale(oof, Y_tr, lam_best)

    params = pd.DataFrame([
        dict(param="s_global", value=s_global),
        dict(param="w_blend", value=w_blend),
        dict(param="lambda_gene", value=lam_best),
    ])
    params.to_csv(PARAM_JSON, index=False)
    print("\n=== frozen parameters (fitted on OOF only) ===")
    print(params.to_string(index=False))
    print(f"gene scale: median={np.median(s_gene_oof):.3f} q10={np.quantile(s_gene_oof,0.10):.3f} "
          f"q90={np.quantile(s_gene_oof,0.90):.3f}")
    pd.DataFrame(blend_rows, columns=["w_aivc", "oof_top20", "oof_mean_r"]).to_csv(
        os.path.join(OUT_DIR, "exp49_blend_curve.csv"), index=False)

    # ---------- apply ONCE to published test predictions ----------
    P0 = np.load(os.path.join(PRED_DIR, "pred_AIVC_v1.1_Ensemble.npy")).astype(np.float64)
    Pr = np.load(os.path.join(PRED_DIR, "pred_Linear_Ridge.npy")).astype(np.float64)

    Pblend = w_blend * P0 + (1 - w_blend) * Pr
    s_blend_global = float((Pblend * Y_ev).sum() / ((Pblend * Pblend).sum() + 1e-12))  # NOT used; kept for logging only

    arms = {
        "A0 original": P0,
        "A1 global scale": P0 * s_global,
        "A2 blend": Pblend,
        "A3 blend + global scale": Pblend * s_global,
        "A4 per-gene scale": P0 * s_gene_oof[None, :],
        "A5 blend + global + per-gene": Pblend * s_global * s_gene_oof[None, :],
    }

    rows = []
    for k, v in arms.items():
        rows.append(report(k, Y_ev, v))
    res = pd.DataFrame(rows)
    res["d_top20_vs_A0"] = res.top20_r - res.loc[0, "top20_r"]
    res["d_wmse_vs_A0"] = res.wmse - res.loc[0, "wmse"]
    res["d_meanr_vs_A0"] = res.mean_r - res.loc[0, "mean_r"]
    res.to_csv(OUT_CSV, index=False)

    print("\n=== blind RPE1 (n=2,016), evaluated once ===")
    print(res.round(4).to_string(index=False))

    np.save(os.path.join(OUT_DIR, "pred_AIVC_v1.2_calibrated.npy"),
            (Pblend * s_global).astype(np.float32))
    print(f"\n[exp49] done in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()