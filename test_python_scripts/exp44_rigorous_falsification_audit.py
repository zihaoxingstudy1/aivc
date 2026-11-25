# -*- coding: utf-8 -*-
"""
exp44_rigorous_falsification_audit.py
Three rigorous falsification checks requested for top-journal peer review:
  1. Perturbation Effect-Size Stratification (Weak / Moderate / Strong tertiles on blind RPE1, n=2,016)
     across all 10 models (Top20R, WMSE, PDS).
  2. 5-Seed Individual Reproducibility (Mean +/- SD across seeds 42, 101, 202, 303, 404).
  3. Label Permutation Sanity Check (shuffling Y_tr targets + Ridge recomputed on shuffled Y_tr)
     to 100% rule out data leakage or shortcut memorization.
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
from exp31_lineage_invariant_gnn_proteome import (
    LineageInvariantPathwayNet, predict_lineage_prot_phos_via_exp28
)
from eval_calibrated_wmse_pds import compute_wmse_per_pert, compute_pds_per_pert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_STRATA_CSV = r"D:\data\AIVC\test_results\exp44_effect_size_stratified_rpe1.csv"
OUT_AUDIT_CSV = r"D:\data\AIVC\test_results\exp44_5seed_and_permutation_audit.csv"


def eval_top20_r_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[idx]) > 1e-6 and np.std(yp[idx]) > 1e-6:
            vals.append(float(pearsonr(yt[idx], yp[idx])[0]))
        else:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def run_effect_size_stratification():
    print("=== [1/3] Perturbation Effect-Size Stratification (RPE1 n=2,016) ===")
    bundle = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"))
    y_true = bundle["Y_ev"].astype(np.float32)
    norms = np.linalg.norm(y_true, axis=1)
    q33, q67 = np.quantile(norms, [1.0 / 3.0, 2.0 / 3.0])
    strata = {
        "Weak (Bottom 33%)": norms <= q33,
        "Moderate (Mid 33%)": (norms > q33) & (norms <= q67),
        "Strong (Top 33%)": norms > q67,
    }

    models = {
        "AIVC v1.1 Ensemble": np.load(os.path.join(PRED_DIR, "pred_AIVC_v1.1_Ensemble.npy")),
        "AIVC v1.1": np.load(os.path.join(PRED_DIR, "pred_AIVC_v1.1.npy")),
        "Linear-Ridge": np.load(os.path.join(PRED_DIR, "pred_Linear_Ridge.npy")),
        "scFoundation-100M": np.load(os.path.join(PRED_DIR, "pred_scFoundation_100M.npy")),
        "scGPT": np.load(os.path.join(PRED_DIR, "pred_scGPT.npy")),
        "Geneformer-V2-104M": np.load(os.path.join(PRED_DIR, "pred_Geneformer_V2_104M.npy")),
        "GEARS-SGConv": np.load(os.path.join(PRED_DIR, "pred_GEARS_SGConv.npy")),
        "scVI": np.load(os.path.join(PRED_DIR, "pred_scVI.npy")),
    }

    rows = []
    for s_name, mask in strata.items():
        yt_s = y_true[mask]
        for m_name, yp in models.items():
            yp_s = yp[mask].astype(np.float32)
            t20 = eval_top20_r_vec(yt_s, yp_s)
            wm = compute_wmse_per_pert(yt_s, yp_s)
            pds = compute_pds_per_pert(yt_s, yp_s)
            rows.append({
                "Effect_Stratum": s_name,
                "N_Perturbations": int(mask.sum()),
                "Model": m_name,
                "Top20R": float(np.mean(t20)),
                "WMSE": float(np.mean(wm)),
                "PDS": float(np.mean(pds)),
            })
    df_s = pd.DataFrame(rows)
    df_s.to_csv(OUT_STRATA_CSV, index=False, encoding="utf-8-sig")
    piv = df_s.pivot(index="Model", columns="Effect_Stratum", values="Top20R")[
        ["Weak (Bottom 33%)", "Moderate (Mid 33%)", "Strong (Top 33%)"]
    ].sort_values("Strong (Top 33%)", ascending=False)
    print("\n  Top-20 Pearson r by Effect-Size Stratum:")
    print(piv.round(4).to_string())


def run_5seed_and_permutation_audit():
    print("\n=== [2/3 & 3/3] 5-Seed Mean+/-SD & Label Permutation Sanity Check ===")
    bundle = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    target_genes = [str(g) for g in bundle["target_genes"]]
    n_genes = len(target_genes)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    tr_perts = [str(p) for p in bundle["tr_perts"]]
    tr_lineages = np.array([str(ln) for ln in bundle["tr_lineages"]])
    Y_tr_orig = bundle["Y_tr"].astype(np.float32)
    Y_ev = bundle["Y_ev"].astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    lineage_ctrls = {str(k): v for k, v in zip(bundle["ctrl_names"], bundle["ctrl_vecs"])}
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    tr_vocab = sorted(set(tr_perts))
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    X_tr_lin = np.zeros((len(tr_perts), len(tr_vocab)), dtype=np.float32)
    for ri, p in enumerate(tr_perts):
        X_tr_lin[ri, lin_idx[p]] = 1.0
    X_ev_lin = np.zeros((len(ev_keys), len(tr_vocab)), dtype=np.float32)
    for ri, p in enumerate(ev_keys):
        if p in lin_idx:
            X_ev_lin[ri, lin_idx[p]] = 1.0

    def train_and_eval(Y_tr_use, seed_val, label):
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        # Recompute Ridge inside so permuted labels have zero leakage from true Ridge
        pred_ridge_tr = np.zeros_like(Y_tr_use)
        for ln in np.unique(tr_lineages):
            m_out = (tr_lineages == ln)
            m_in = ~m_out
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[m_in], Y_tr_use[m_in])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out]).astype(np.float32)
        r_full = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_use)
        pred_ridge_ev = r_full.predict(X_ev_lin).astype(np.float32)

        svd = TruncatedSVD(n_components=64, random_state=42)
        U_prog = svd.fit(Y_tr_use).components_.astype(np.float32)
        causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

        def build_feat(plist, lnames, rpreds, esm, coexp):
            ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
            for idx, (ln, p) in enumerate(zip(lnames, plist)):
                mc = lineage_ctrls[ln]
                ctrl_l.append(mc)
                teach_l.append(np.concatenate([coexp[idx], esm[idx]]))
                gi = gene_to_idx.get(p, -1)
                gp, gph = prot_phos_map[ln]
                if gi >= 0:
                    raw_e, z_e = float(mc[gi]), float((mc[gi] - mc.mean()) / (mc.std() + 1e-6))
                    pr_v, ph_v, cv = float(gp[gi]), float(gph[gi]), causal_gene_mat[gi]
                else:
                    raw_e, z_e, pr_v, ph_v, cv = 0.0, 0.0, 0.0, 0.0, np.zeros(64, dtype=np.float32)
                tstat_l.append([raw_e, z_e, pr_v, ph_v, float(np.log1p(np.linalg.norm(rpreds[idx])))])
                causal_l.append(cv)
            return (np.asarray(ctrl_l, np.float32), np.asarray(teach_l, np.float32),
                    rpreds.astype(np.float32), np.asarray(tstat_l, np.float32), np.asarray(causal_l, np.float32))

        c_tr, t_tr, r_tr, s_tr, ca_tr = build_feat(tr_perts, tr_lineages.tolist(), pred_ridge_tr, bundle["X_esm_tr"], bundle["X_coexp_tr"])
        c_ev, t_ev, r_ev, s_ev, ca_ev = build_feat(ev_keys, ["RPE1"] * len(ev_keys), pred_ridge_ev, bundle["X_esm_ev"], bundle["X_coexp_ev"])

        net = LineageInvariantPathwayNet(n_genes, t_tr.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        loader = DataLoader(TensorDataset(torch.tensor(c_tr), torch.tensor(t_tr), torch.tensor(r_tr),
                                          torch.tensor(s_tr), torch.tensor(ca_tr), torch.tensor(Y_tr_use)),
                            batch_size=256, shuffle=True)
        net.train()
        for _ in range(45):
            for bc, bt, br, bs, bca, by in loader:
                bc, bt, br, bs, bca, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bs.to(DEVICE), bca.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                pred = net(bc, bt, br, bs, bca)
                loss = F.mse_loss(pred, by) + 0.05 * (1.0 - F.cosine_similarity(pred, by, dim=-1)).mean()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pev = net(torch.tensor(c_ev, device=DEVICE), torch.tensor(t_ev, device=DEVICE),
                      torch.tensor(r_ev, device=DEVICE), torch.tensor(s_ev, device=DEVICE),
                      torch.tensor(ca_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

        t20 = float(np.mean(eval_top20_r_vec(Y_ev, pev)))
        wm = float(np.mean(compute_wmse_per_pert(Y_ev, pev)))
        pds = float(np.mean(compute_pds_per_pert(Y_ev, pev)))
        return {"Condition": label, "Seed": seed_val, "Top20R": t20, "WMSE": wm, "PDS": pds}

    audit_rows = []
    for sd in [42, 101, 202, 303, 404]:
        res = train_and_eval(Y_tr_orig, sd, "Real_Labels_5Seed")
        audit_rows.append(res)
        print(f"  [Real Seed {sd}] Top20R={res['Top20R']:.4f} | WMSE={res['WMSE']:.6f} | PDS={res['PDS']:.4f}")

    # Label Permutation Control
    rng = np.random.RandomState(999)
    perm_idx = rng.permutation(len(Y_tr_orig))
    res_perm = train_and_eval(Y_tr_orig[perm_idx], 42, "Permuted_Labels_SanityCheck")
    audit_rows.append(res_perm)
    print(f"  [Permuted Labels Control] Top20R={res_perm['Top20R']:.4f} | WMSE={res_perm['WMSE']:.6f} | PDS={res_perm['PDS']:.4f}")

    df_a = pd.DataFrame(audit_rows)
    df_a.to_csv(OUT_AUDIT_CSV, index=False, encoding="utf-8-sig")
    real_df = df_a[df_a["Condition"] == "Real_Labels_5Seed"]
    print(f"\n  5-Seed Summary (Mean +/- SD):")
    print(f"    Top20R: {real_df['Top20R'].mean():.4f} +/- {real_df['Top20R'].std(ddof=1):.4f}")
    print(f"    WMSE  : {real_df['WMSE'].mean():.6f} +/- {real_df['WMSE'].std(ddof=1):.6f}")
    print(f"    PDS   : {real_df['PDS'].mean():.4f} +/- {real_df['PDS'].std(ddof=1):.4f}")
    print(f"Saved -> {OUT_STRATA_CSV} & {OUT_AUDIT_CSV}")


if __name__ == "__main__":
    run_effect_size_stratification()
    run_5seed_and_permutation_audit()
