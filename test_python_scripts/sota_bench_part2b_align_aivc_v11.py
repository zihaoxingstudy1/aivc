# -*- coding: utf-8 -*-
"""
sota_bench_part2b_align_aivc_v11.py
Re-generate pred_AIVC_v1.1.npy and pred_AIVC_v1.1_Ensemble.npy using the EXACT Exp31 loss
(MSE + 0.05 * cosine_loss, lr=1e-3, wd=1e-3, bs=256, epochs=45) on rpe1_ood_bundle.npz.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
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

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
PRIORS_DIR = r"D:\data\AIVC\data\priors"


def main():
    t0 = time.time()
    bundle = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    target_genes = [str(g) for g in bundle["target_genes"]]
    n_genes = len(target_genes)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    tr_perts = [str(p) for p in bundle["tr_perts"]]
    tr_lineages = np.array([str(ln) for ln in bundle["tr_lineages"]])
    Y_tr = bundle["Y_tr"]

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

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

    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U_prog = svd.fit(Y_tr).components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

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

    ctrl_tr, teach_tr, r_tr, tstat_tr, caus_tr = build_features(tr_perts, tr_lineages.tolist(), pred_ridge_tr, bundle["X_esm_tr"], bundle["X_coexp_tr"])
    ctrl_ev, teach_ev, r_ev, tstat_ev, caus_ev = build_features(ev_keys, ["RPE1"] * len(ev_keys), pred_ridge_ev, bundle["X_esm_ev"], bundle["X_coexp_ev"])

    seeds = [42, 101, 202, 303, 404]
    preds = []
    for sd_val in seeds:
        torch.manual_seed(sd_val)
        np.random.seed(sd_val)
        net = LineageInvariantPathwayNet(n_genes, teach_tr.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(ctrl_tr), torch.tensor(teach_tr), torch.tensor(r_tr),
                torch.tensor(tstat_tr), torch.tensor(caus_tr), torch.tensor(Y_tr),
            ),
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
            pev = net(
                torch.tensor(ctrl_ev, device=DEVICE), torch.tensor(teach_ev, device=DEVICE),
                torch.tensor(r_ev, device=DEVICE), torch.tensor(tstat_ev, device=DEVICE),
                torch.tensor(caus_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        preds.append(pev)
        print(f"  Finished seed={sd_val}")

    np.save(os.path.join(OUT_DIR, "pred_AIVC_v1.1.npy"), preds[0])
    np.save(os.path.join(OUT_DIR, "pred_AIVC_v1.1_Ensemble.npy"), np.mean(preds, axis=0).astype(np.float32))
    print(f"Updated AIVC v1.1 & Ensemble predictions in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
