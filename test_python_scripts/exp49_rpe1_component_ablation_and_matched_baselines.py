# -*- coding: utf-8 -*-
"""
exp49_rpe1_component_ablation_and_matched_baselines.py
Answers: WHAT drives AIVC's gain over Linear-Ridge on blind RPE1 (n=2,016)?
Architecture (exp31 LineageInvariantPathwayNet): pred = gate * ridge_scale * x_ridge + 0.5 * (prog_head(h) @ U_prog),
h = [ctrl_enc(control profile), teach_enc([coexp(64), ESM2(640)]), causal_enc(OmniPath footprint), x_tstat(5)].
Arms (all 5 seeds, same data/split/optimizer as the paper; only the listed input is zeroed / loss changed):
  full_signed, no_prior, no_ctrl, no_esm2, no_coexp, no_teach_all, no_5omics_state (prot/phos zeroed),
  no_tstat_all, all_inputs_zero (only Ridge anchor + learned scale/constant), mse_only (no cosine loss)
Non-neural, information-matched baselines:
  ridge_only, ridge_affine_recalibration (Y ~ s*r + b_g), residual_ridge (Ridge on [ctrl,teach,tstat] -> Y - r)
Paired perturbation-level bootstrap CI (2,000 resamples) of the Top-20 r difference vs full_signed.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp31_lineage_invariant_gnn_proteome import LineageInvariantPathwayNet, predict_lineage_prot_phos_via_exp28
from eval_calibrated_wmse_pds import compute_wmse_per_pert, compute_pds_per_pert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp49_rpe1_component_ablation.csv"
OUT_PAIRED = r"D:\data\AIVC\test_results\exp49_rpe1_paired_bootstrap.csv"
SEEDS = [42, 101, 202, 303, 404]
N_BOOT = 2000


def per_pert_top20(yt, yp):
    out = np.zeros(len(yt), dtype=np.float32)
    for i in range(len(yt)):
        idx = np.argsort(-np.abs(yt[i]))[:20]
        if np.std(yt[i][idx]) > 1e-6 and np.std(yp[i][idx]) > 1e-6:
            out[i] = pearsonr(yt[i][idx], yp[i][idx])[0]
    return out


def summarize(yt, yp, weak_mask):
    t20 = per_pert_top20(yt, yp)
    pa = []
    for i in range(len(yt)):
        if np.std(yt[i]) > 1e-6 and np.std(yp[i]) > 1e-6:
            pa.append(pearsonr(yt[i], yp[i])[0])
    dd = []
    for i in range(len(yt)):
        idx = np.argsort(-np.abs(yt[i]))[:20]
        dd.append((np.sign(yt[i][idx]) == np.sign(yp[i][idx])).mean())
    d = {"MeanR": float(np.mean(pa)), "Top20R": float(t20.mean()), "DirAcc": float(np.mean(dd)),
         "MSE": float(np.mean((yt - yp) ** 2)), "WMSE": float(np.mean(compute_wmse_per_pert(yt, yp))),
         "PDS": float(np.mean(compute_pds_per_pert(yt, yp))), "Weak_Top20R": float(t20[weak_mask].mean())}
    return d, t20


def main():
    b = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    genes = [str(g) for g in b["target_genes"]]
    G = len(genes)
    g2i = {g: i for i, g in enumerate(genes)}
    ev_keys = [str(k) for k in b["ev_keys"]]
    tr_perts = [str(p) for p in b["tr_perts"]]
    tr_lin = np.array([str(x) for x in b["tr_lineages"]])
    Y_tr = b["Y_tr"].astype(np.float32)
    Y_ev = b["Y_ev"].astype(np.float32)
    weak = np.linalg.norm(Y_ev, axis=1) <= np.quantile(np.linalg.norm(Y_ev, axis=1), 1 / 3)

    bip = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A = (bip["A_pos"] - bip["A_neg"]).astype(np.float32)
    ctrls = {str(k): v for k, v in zip(b["ctrl_names"], b["ctrl_vecs"])}
    pp = predict_lineage_prot_phos_via_exp28(ctrls, genes)

    r_ev = np.load(os.path.join(OUT_DIR, "pred_Linear_Ridge.npy")).astype(np.float32)
    vocab = sorted(set(tr_perts))
    vi = {t: i for i, t in enumerate(vocab)}
    X_lin = np.zeros((len(tr_perts), len(vocab)), np.float32)
    for ri, p in enumerate(tr_perts):
        X_lin[ri, vi[p]] = 1.0
    r_tr = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lin):
        mo = tr_lin == ln
        r_tr[mo] = Ridge(alpha=50.0).fit(X_lin[~mo], Y_tr[~mo]).predict(X_lin[mo]).astype(np.float32)

    U = TruncatedSVD(n_components=64, random_state=42).fit(Y_tr).components_.astype(np.float32)
    causal_mat = ((A + 0.5 * (A @ A)) @ U.T).astype(np.float32)
    n_coexp = b["X_coexp_tr"].shape[1]

    def feats(plist, lnames, rp, esm, coexp):
        c, t, s, ca = [], [], [], []
        for k, (ln, p) in enumerate(zip(lnames, plist)):
            mc = ctrls[ln]
            c.append(mc)
            t.append(np.concatenate([coexp[k], esm[k]]))
            gi = g2i.get(p, -1)
            gp, gph = pp[ln]
            if gi >= 0:
                s.append([float(mc[gi]), float((mc[gi] - mc.mean()) / (mc.std() + 1e-6)), float(gp[gi]), float(gph[gi]), float(np.log1p(np.linalg.norm(rp[k])))])
                ca.append(causal_mat[gi])
            else:
                s.append([0, 0, 0, 0, float(np.log1p(np.linalg.norm(rp[k])))])
                ca.append(np.zeros(64, np.float32))
        return [np.asarray(x, np.float32) for x in (c, t, s, ca)]

    ctr, ttr, str_, catr = feats(tr_perts, tr_lin.tolist(), r_tr, b["X_esm_tr"], b["X_coexp_tr"])
    cev, tev, sev, caev = feats(ev_keys, ["RPE1"] * len(ev_keys), r_ev, b["X_esm_ev"], b["X_coexp_ev"])

    def mod(name, c, t, s, ca):
        c, t, s, ca = c.copy(), t.copy(), s.copy(), ca.copy()
        if name in ("no_prior", "all_inputs_zero"):
            ca[:] = 0
        if name in ("no_ctrl", "all_inputs_zero"):
            c[:] = 0
        if name in ("no_esm2", "no_teach_all", "all_inputs_zero"):
            t[:, n_coexp:] = 0
        if name in ("no_coexp", "no_teach_all", "all_inputs_zero"):
            t[:, :n_coexp] = 0
        if name == "no_5omics_state":
            s[:, 2:4] = 0
        if name in ("no_tstat_all", "all_inputs_zero"):
            s[:, :] = 0
        return c, t, s, ca

    arms = ["full_signed", "no_prior", "no_ctrl", "no_esm2", "no_coexp", "no_teach_all", "no_5omics_state", "no_tstat_all", "all_inputs_zero", "mse_only"]
    rows, vecs = [], {}
    for arm in arms:
        c1, t1, s1, ca1 = mod(arm, ctr, ttr, str_, catr)
        c2, t2, s2, ca2 = mod(arm, cev, tev, sev, caev)
        preds, seed_t20 = [], []
        for sd in SEEDS:
            torch.manual_seed(sd)
            np.random.seed(sd)
            net = LineageInvariantPathwayNet(G, t1.shape[1], 64, U, stat_dim=5, hid=128).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            dl = DataLoader(TensorDataset(*[torch.tensor(x) for x in (c1, t1, r_tr, s1, ca1, Y_tr)]), batch_size=256, shuffle=True)
            net.train()
            for _ in range(45):
                for bc, bt, br, bs, bca, by in dl:
                    bc, bt, br, bs, bca, by = [x.to(DEVICE) for x in (bc, bt, br, bs, bca, by)]
                    opt.zero_grad()
                    pr = net(bc, bt, br, bs, bca)
                    loss = F.mse_loss(pr, by)
                    if arm != "mse_only":
                        loss = loss + 0.05 * (1.0 - F.cosine_similarity(pr, by, dim=-1)).mean()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                pe = net(*[torch.tensor(x, device=DEVICE) for x in (c2, t2, r_ev, s2, ca2)]).cpu().numpy().astype(np.float32)
            preds.append(pe)
            seed_t20.append(per_pert_top20(Y_ev, pe).mean())
        pens = np.mean(preds, axis=0).astype(np.float32)
        m, v = summarize(Y_ev, pens, weak)
        m.update({"Arm": arm, "Seed5_Top20R_Mean": float(np.mean(seed_t20)), "Seed5_Top20R_SD": float(np.std(seed_t20, ddof=1))})
        rows.append(m)
        vecs[arm] = v
        print(f"[{arm:16s}] Ens Top20={m['Top20R']:.4f} Weak={m['Weak_Top20R']:.4f} PDS={m['PDS']:.4f} MSE={m['MSE']:.5f} | 5seed {m['Seed5_Top20R_Mean']:.4f}+/-{m['Seed5_Top20R_SD']:.4f}", flush=True)

    # ---------------- information-matched non-neural baselines ----------------
    m, v = summarize(Y_ev, r_ev, weak); m["Arm"] = "ridge_only"; m["Seed5_Top20R_Mean"] = m["Top20R"]; m["Seed5_Top20R_SD"] = 0.0; rows.append(m); vecs["ridge_only"] = v
    rc, yc = r_tr - r_tr.mean(0), Y_tr - Y_tr.mean(0)
    s_glob = float((rc * yc).sum() / (rc * rc).sum())
    bias = Y_tr.mean(0) - s_glob * r_tr.mean(0)
    m, v = summarize(Y_ev, s_glob * r_ev + bias, weak); m["Arm"] = f"ridge_affine_recal(s={s_glob:.3f})"; m["Seed5_Top20R_Mean"] = m["Top20R"]; m["Seed5_Top20R_SD"] = 0.0; rows.append(m); vecs[m["Arm"]] = v
    Xa = np.hstack([ctr, ttr, str_]); Xe = np.hstack([cev, tev, sev])
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xa, Xe = (Xa - mu) / sd, (Xe - mu) / sd
    for alpha in [100.0, 1000.0, 10000.0]:
        rr = Ridge(alpha=alpha).fit(Xa, Y_tr - r_tr)
        m, v = summarize(Y_ev, r_ev + rr.predict(Xe).astype(np.float32), weak)
        m["Arm"] = f"residual_ridge(alpha={alpha:g})"; m["Seed5_Top20R_Mean"] = m["Top20R"]; m["Seed5_Top20R_SD"] = 0.0
        rows.append(m); vecs[m["Arm"]] = v
        print(f"[{m['Arm']:28s}] Top20={m['Top20R']:.4f} Weak={m['Weak_Top20R']:.4f} PDS={m['PDS']:.4f}", flush=True)

    df = pd.DataFrame(rows)[["Arm", "MeanR", "Top20R", "DirAcc", "MSE", "WMSE", "PDS", "Weak_Top20R", "Seed5_Top20R_Mean", "Seed5_Top20R_SD"]]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- paired perturbation-level bootstrap vs full_signed ----------------
    rng = np.random.RandomState(0)
    idxs = rng.randint(0, len(Y_ev), size=(N_BOOT, len(Y_ev)))
    prow = []
    ref = vecs["full_signed"]
    for arm, v in vecs.items():
        if arm == "full_signed":
            continue
        d = v - ref
        bs_ = d[idxs].mean(1)
        prow.append({"Arm_vs_full_signed": arm, "Delta_Top20R_(arm-full)": float(d.mean()),
                     "CI95_lo": float(np.percentile(bs_, 2.5)), "CI95_hi": float(np.percentile(bs_, 97.5))})
    dp = pd.DataFrame(prow)
    dp.to_csv(OUT_PAIRED, index=False, encoding="utf-8-sig")
    print("\n" + df.round(4).to_string(index=False))
    print("\nPaired bootstrap (Top-20 r difference of each arm minus full_signed):")
    print(dp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
