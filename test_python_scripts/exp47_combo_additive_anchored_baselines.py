# -*- coding: utf-8 -*-
"""
exp47_combo_additive_anchored_baselines.py
Fairness check for Norman2019 unseen double-gene combinations (104 train -> 27 unseen).
AIVC_Combo_Epistasis receives the additive anchor (dA + dB) as an INPUT; the Linear-Ridge / Static-MLP /
foundation-model heads in the original benchmark did not. Here every baseline gets the same information:
  (1) Additive                         : dA + dB
  (2) Additive x fitted gain           : s * (dA + dB), s fit on training combos (global scalar shrinkage)
  (3) Additive + Ridge residual        : (dA + dB) + Ridge([eA+eB, eA*eB])  (symmetric ESM-2 features)
  (4) Additive x gain + Ridge residual : s * (dA + dB) + Ridge residual on the remaining error
Metrics: Mean r, Top-20 r, DirAcc, MSE, non-additive epistasis r (corr of y - (dA+dB) with prediction - (dA+dB)).
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp3_triteacher_distill import load_esm2_gene_embeddings

BUNDLE = r"D:\data\AIVC\test_results\sota_norman_preds\norman_ood_bundle.npz"
ESM = r"D:\data\AIVC\data\target_esm2_150m.npz"
OUT = r"D:\data\AIVC\test_results\exp47_combo_additive_anchored_baselines.csv"


def metrics(yt, yp, yadd):
    pa, p20, dd = [], [], []
    for i in range(len(yt)):
        if np.std(yt[i]) > 1e-6 and np.std(yp[i]) > 1e-6:
            pa.append(pearsonr(yt[i], yp[i])[0])
        idx = np.argsort(-np.abs(yt[i]))[:20]
        if np.std(yt[i][idx]) > 1e-6 and np.std(yp[i][idx]) > 1e-6:
            p20.append(pearsonr(yt[i][idx], yp[i][idx])[0])
        dd.append((np.sign(yt[i][idx]) == np.sign(yp[i][idx])).mean())
    res_t = (yt - yadd).ravel()
    res_p = (yp - yadd).ravel()
    epi = float(pearsonr(res_t, res_p)[0]) if np.std(res_p) > 1e-9 else 0.0
    return float(np.mean(pa)), float(np.mean(p20)), float(np.mean(dd)), float(np.mean((yt - yp) ** 2)), epi


def main():
    b = np.load(BUNDLE, allow_pickle=True)
    tr_s = [str(g).upper() for g in b["tr_s"]]
    ev_s = [str(g).upper() for g in b["ev_s"]]
    singles = {g: v for g, v in zip(tr_s, b["Ys_tr"])}
    singles.update({g: v for g, v in zip(ev_s, b["Ys_ev"])})
    esm = {k.upper(): v for k, v in load_esm2_gene_embeddings(ESM).items()}
    d_esm = len(next(iter(esm.values())))

    def build(gA, gB, Y):
        add, feat, keep = [], [], []
        for i, (a, c) in enumerate(zip(gA, gB)):
            a, c = str(a).upper(), str(c).upper()
            if a in singles and c in singles:
                add.append(singles[a] + singles[c])
                ea = esm.get(a, np.zeros(d_esm, np.float32)); ec = esm.get(c, np.zeros(d_esm, np.float32))
                feat.append(np.concatenate([ea + ec, ea * ec]))
                keep.append(i)
        return np.asarray(add, np.float32), np.asarray(feat, np.float32), Y[keep]

    Xa_tr, F_tr, Y_tr = build(b["tr_c_gA"], b["tr_c_gB"], b["Yc_tr"].astype(np.float32))
    Xa_ev, F_ev, Y_ev = build(b["ev_c_gA"], b["ev_c_gB"], b["Yc_ev"].astype(np.float32))
    print(f"train combos usable={len(Y_tr)}/{len(b['Yc_tr'])}, eval combos usable={len(Y_ev)}/{len(b['Yc_ev'])}")

    s = float((Xa_tr * Y_tr).sum() / (Xa_tr * Xa_tr).sum())
    print(f"Fitted global gain on additive anchor s = {s:.4f}")
    rows = []

    def add_row(name, pred):
        m = metrics(Y_ev, pred, Xa_ev)
        rows.append({"Model": name, "MeanR": m[0], "Top20R": m[1], "DirAcc": m[2], "MSE": m[3], "EpistasisR": m[4]})

    add_row("Additive (dA+dB)", Xa_ev)
    add_row(f"Additive x fitted gain (s={s:.3f})", s * Xa_ev)
    for alpha in [10.0, 100.0, 1000.0]:
        r1 = Ridge(alpha=alpha).fit(F_tr, Y_tr - Xa_tr)
        add_row(f"Additive + Ridge-residual (alpha={alpha:g})", Xa_ev + r1.predict(F_ev))
        r2 = Ridge(alpha=alpha).fit(F_tr, Y_tr - s * Xa_tr)
        add_row(f"Gain-scaled additive + Ridge-residual (alpha={alpha:g})", s * Xa_ev + r2.predict(F_ev))
    ref = pd.read_csv(r"D:\data\AIVC\test_results\sota_benchmark_norman_combo.csv")
    for _, r in ref[ref["model"].str.contains("AIVC")].iterrows():
        rows.append({"Model": "AIVC_Combo_Epistasis_Ens5 (original benchmark)", "MeanR": r["mean_pearson"], "Top20R": r["top20_pearson"],
                     "DirAcc": r["dir_acc_top20"], "MSE": r["mse"], "EpistasisR": r["epistasis_r"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
