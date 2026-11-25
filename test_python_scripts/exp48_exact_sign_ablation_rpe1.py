# -*- coding: utf-8 -*-
"""
exp48_exact_sign_ablation_rpe1.py
Exact reproduction of AIVC v1.1 (sota_bench_part2b_align_aivc_v11.py) with 3-arm OmniPath prior ablation:
  1. signed   : A_signed = A_pos - A_neg (exact paper model)
  2. unsigned : A_unsign = A_pos + A_neg (sign stripped, topology kept)
  3. none     : A_zero   = 0             (OmniPath prior completely removed)
Evaluates exact 5-seed Ensemble (42, 101, 202, 303, 404) + per-seed Mean +/- SD on:
  - All blind RPE1 perturbations (n=2,016)
  - Edge-covered subset (perturbations with >=1 OmniPath edge)
  - Weak effect-size tertile (Bottom 33%, n=672)
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
OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp48_exact_sign_ablation_rpe1.csv"


def eval_metrics(y_true, y_pred):
    pa, p20, d20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            pa.append(float(pearsonr(yt, yp)[0]))
        idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[idx]) > 1e-6 and np.std(yp[idx]) > 1e-6:
            p20.append(float(pearsonr(yt[idx], yp[idx])[0]))
        d20.append(float((np.sign(yt[idx]) == np.sign(yp[idx])).mean()))
    wmse = float(np.mean(compute_wmse_per_pert(y_true, y_pred)))
    pds = float(np.mean(compute_pds_per_pert(y_true, y_pred)))
    return float(np.mean(pa)), float(np.mean(p20)), float(np.mean(d20)), float(np.mean((y_true - y_pred) ** 2)), wmse, pds


def main():
    bundle = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    target_genes = [str(g) for g in bundle["target_genes"]]
    n_genes = len(target_genes)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    tr_perts = [str(p) for p in bundle["tr_perts"]]
    tr_lineages = np.array([str(ln) for ln in bundle["tr_lineages"]])
    Y_tr = bundle["Y_tr"].astype(np.float32)
    Y_ev = bundle["Y_ev"].astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_pos = bipartite["A_pos"].astype(np.float32)
    A_neg = bipartite["A_neg"].astype(np.float32)

    # Check exact edge coverage on 2,016 blind RPE1 perturbations
    has_edge = np.zeros(len(ev_keys), dtype=bool)
    for i, p in enumerate(ev_keys):
        gi = gene_to_idx.get(p, -1)
        if gi >= 0 and (np.abs(A_pos[gi]).sum() + np.abs(A_neg[gi]).sum()) > 0:
            has_edge[i] = True
    norms = np.linalg.norm(Y_ev, axis=1)
    q33 = np.quantile(norms, 1.0 / 3.0)
    weak_mask = norms <= q33
    print(f"Blind RPE1 n={len(ev_keys)} | With OmniPath edges: {has_edge.sum()} ({100*has_edge.mean():.1f}%) | Weak tertile: {weak_mask.sum()}")

    lineage_ctrls = {str(k): v for k, v in zip(bundle["ctrl_names"], bundle["ctrl_vecs"])}
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    pred_ridge_ev = np.load(os.path.join(OUT_DIR, "pred_Linear_Ridge.npy"))
    tr_vocab = sorted(set(tr_perts))
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    X_tr_lin = np.zeros((len(tr_perts), len(tr_vocab)), dtype=np.float32)
    for ri, p in enumerate(tr_perts):
        X_tr_lin[ri, lin_idx[p]] = 1.0
    pred_ridge_tr = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lineages):
        m_out = (tr_lineages == ln)
        m_in = ~m_out
        if m_in.sum() > 0:
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[m_in], Y_tr[m_in])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out]).astype(np.float32)

    svd = TruncatedSVD(n_components=64, random_state=42)
    U_prog = svd.fit(Y_tr).components_.astype(np.float32)

    arms = {
        "signed (A_pos - A_neg)": A_pos - A_neg,
        "unsigned (A_pos + A_neg)": A_pos + A_neg,
        "none (zeros)": np.zeros_like(A_pos),
    }

    rows = []
    for arm_name, A_mat in arms.items():
        causal_gene_mat = ((A_mat + 0.5 * (A_mat @ A_mat)) @ U_prog.T).astype(np.float32)

        def build_features(plist, lnames, ridge_preds, esm_mat, coexp_mat):
            ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
            for idx, (ln, p) in enumerate(zip(lnames, plist)):
                mc = lineage_ctrls[ln]
                ctrl_l.append(mc)
                teach_l.append(np.concatenate([coexp_mat[idx], esm_mat[idx]]))
                gi = gene_to_idx.get(p, -1)
                g_prot, g_phos = prot_phos_map[ln]
                if gi >= 0:
                    raw_e = float(mc[gi])
                    z_e = float((mc[gi] - mc.mean()) / (mc.std() + 1e-6))
                    pr_v, ph_v = float(g_prot[gi]), float(g_phos[gi])
                    c_vec = causal_gene_mat[gi]
                else:
                    raw_e, z_e, pr_v, ph_v = 0.0, 0.0, 0.0, 0.0
                    c_vec = np.zeros(64, dtype=np.float32)
                rn = float(np.log1p(np.linalg.norm(ridge_preds[idx])))
                tstat_l.append([raw_e, z_e, pr_v, ph_v, rn])
                causal_l.append(c_vec)
            return (
                np.asarray(ctrl_l, dtype=np.float32),
                np.asarray(teach_l, dtype=np.float32),
                ridge_preds.astype(np.float32),
                np.asarray(tstat_l, dtype=np.float32),
                np.asarray(causal_l, dtype=np.float32),
            )

        c_tr, t_tr, r_tr, s_tr, ca_tr = build_features(tr_perts, tr_lineages.tolist(), pred_ridge_tr, bundle["X_esm_tr"], bundle["X_coexp_tr"])
        c_ev, t_ev, r_ev, s_ev, ca_ev = build_features(ev_keys, ["RPE1"] * len(ev_keys), pred_ridge_ev, bundle["X_esm_ev"], bundle["X_coexp_ev"])

        preds = []
        seed_t20 = []
        for sd_val in [42, 101, 202, 303, 404]:
            torch.manual_seed(sd_val)
            np.random.seed(sd_val)
            net = LineageInvariantPathwayNet(n_genes, t_tr.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            loader = DataLoader(
                TensorDataset(torch.tensor(c_tr), torch.tensor(t_tr), torch.tensor(r_tr),
                              torch.tensor(s_tr), torch.tensor(ca_tr), torch.tensor(Y_tr)),
                batch_size=256, shuffle=True,
            )
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
            preds.append(pev)
            seed_t20.append(eval_metrics(Y_ev, pev)[1])

        p_ens = np.mean(preds, axis=0).astype(np.float32)
        m_all = eval_metrics(Y_ev, p_ens)
        m_edge = eval_metrics(Y_ev[has_edge], p_ens[has_edge])
        m_weak = eval_metrics(Y_ev[weak_mask], p_ens[weak_mask])
        rows.append({
            "Arm": arm_name,
            "Ens_MeanR": m_all[0], "Ens_Top20R": m_all[1], "Ens_DirAcc": m_all[2],
            "Ens_MSE": m_all[3], "Ens_WMSE": m_all[4], "Ens_PDS": m_all[5],
            "EdgeSubset_Top20R": m_edge[1], "WeakTertile_Top20R": m_weak[1],
            "Seed5_Top20R_Mean": float(np.mean(seed_t20)), "Seed5_Top20R_SD": float(np.std(seed_t20, ddof=1)),
        })
        print(f"  [{arm_name}] All Top20={m_all[1]:.4f} (MeanR={m_all[0]:.4f}, PDS={m_all[5]:.4f}) | EdgeSub={m_edge[1]:.4f} | Weak={m_weak[1]:.4f} | 5-Seed={np.mean(seed_t20):.4f}+/-{np.std(seed_t20, ddof=1):.4f}")

    # Linear-Ridge reference
    r_all = eval_metrics(Y_ev, pred_ridge_ev)
    r_edge = eval_metrics(Y_ev[has_edge], pred_ridge_ev[has_edge])
    r_weak = eval_metrics(Y_ev[weak_mask], pred_ridge_ev[weak_mask])
    rows.append({
        "Arm": "Linear-Ridge",
        "Ens_MeanR": r_all[0], "Ens_Top20R": r_all[1], "Ens_DirAcc": r_all[2],
        "Ens_MSE": r_all[3], "Ens_WMSE": r_all[4], "Ens_PDS": r_all[5],
        "EdgeSubset_Top20R": r_edge[1], "WeakTertile_Top20R": r_weak[1],
        "Seed5_Top20R_Mean": r_all[1], "Seed5_Top20R_SD": 0.0,
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + df.round(4).to_string(index=False))
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
