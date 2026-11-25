# -*- coding: utf-8 -*-
"""
sota_bench_part1_prepare_and_baselines.py
Export the exact 7-Lineage -> Unseen RPE1 (2,016 perturbations) OOD benchmark bundle
and compute Train-Mean, Linear-Ridge, Static-MLP, and AIVC v1.1 (Exp31) predictions.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from bfc import build_gene_coexp_prior

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
os.makedirs(OUT_DIR, exist_ok=True)


class StaticMLPPert(nn.Module):
    def __init__(self, d_in: int, n_genes: int, hid: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, n_genes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    t0 = time.time()
    print(f"=== SOTA Benchmark Part 1: Prepare OOD Bundle & Baselines ({DEVICE}) ===")
    ref_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    ref = sc.read_h5ad(ref_h5ad)
    target_genes = list(ref.var_names)
    n_genes = len(target_genes)

    tr_ref = ref[ref.obs["ood_split"] == "train"]
    ev_ref = ref[ref.obs["ood_split"] == "test"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    mu_rpe1 = _to_dense_array(ev_ref[ev_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    d_rpe1 = _compute_group_deltas(ev_ref[ev_ref.obs["perturbation"] != "control"], "perturbation", mu_rpe1)

    coexp_dict = build_gene_coexp_prior(tr_ref.X, target_genes, prior_dim=64)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    del ref, tr_ref, ev_ref

    lineage_specs = [("K562", mu_k562, d_k562)]
    extra_files = [
        ("HepG2", r"D:\data\AIVC\data\NadigOConner2024_hepg2.h5ad", False),
        ("Jurkat", r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad", False),
        ("iPSC", r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad", True),
        ("Norman", r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad", True),
        ("PrimaryT", r"D:\data\AIVC\data\ShifrutMarson2018.h5ad", True),
        ("THP1", r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad", True),
    ]
    for lname, fpath, sp_only in extra_files:
        if os.path.exists(fpath):
            mc, dd = extract_lineage_deltas(fpath, target_genes, lname, min_cells=10, single_pert_only=sp_only)
            lineage_specs.append((lname, mc, dd))

    tr_samples = []
    lineage_ctrls = {"RPE1": mu_rpe1}
    for lname, mc, dd in lineage_specs:
        lineage_ctrls[lname] = mc
        for p, dv in dd.items():
            if "_" in p or "+" in p:
                continue
            tr_samples.append((lname, p, dv))

    ev_keys = sorted(d_rpe1.keys())
    Y_ev = np.vstack([d_rpe1[k] for k in ev_keys]).astype(np.float32)
    Y_tr = np.vstack([s[2] for s in tr_samples]).astype(np.float32)
    tr_lineages = np.array([s[0] for s in tr_samples])
    tr_perts = np.array([s[1] for s in tr_samples])

    d_esm = len(next(iter(esm2_dict.values())))
    X_esm_tr = np.vstack([esm2_dict.get(p, np.zeros(d_esm, dtype=np.float32)) for p in tr_perts]).astype(np.float32)
    X_esm_ev = np.vstack([esm2_dict.get(p, np.zeros(d_esm, dtype=np.float32)) for p in ev_keys]).astype(np.float32)
    X_coexp_tr = np.vstack([coexp_dict.get(p, np.zeros(64, dtype=np.float32)) for p in tr_perts]).astype(np.float32)
    X_coexp_ev = np.vstack([coexp_dict.get(p, np.zeros(64, dtype=np.float32)) for p in ev_keys]).astype(np.float32)

    bundle_path = os.path.join(OUT_DIR, "rpe1_ood_bundle.npz")
    np.savez_compressed(
        bundle_path,
        target_genes=np.array(target_genes),
        mu_k562=mu_k562,
        mu_rpe1=mu_rpe1,
        ev_keys=np.array(ev_keys),
        Y_ev=Y_ev,
        tr_lineages=tr_lineages,
        tr_perts=tr_perts,
        Y_tr=Y_tr,
        X_esm_tr=X_esm_tr,
        X_esm_ev=X_esm_ev,
        X_coexp_tr=X_coexp_tr,
        X_coexp_ev=X_coexp_ev,
        ctrl_names=np.array(sorted(lineage_ctrls.keys())),
        ctrl_vecs=np.vstack([lineage_ctrls[k] for k in sorted(lineage_ctrls.keys())]).astype(np.float32),
    )
    print(f"Saved OOD bundle -> {bundle_path} (train={len(tr_samples)}, test_rpe1={len(ev_keys)})")

    # 1. Train-Mean Baseline
    mean_vec = Y_tr.mean(axis=0, keepdims=True)
    pred_mean = np.repeat(mean_vec, len(ev_keys), axis=0).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "pred_Train_Mean.npy"), pred_mean)

    # 2. Linear-Ridge Baseline (perturbation one-hot, identical to Exp31/v1.0 Ridge)
    tr_vocab = sorted(set(tr_perts.tolist()))
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(enc_lin(tr_perts), Y_tr)
    pred_ridge = ridge.predict(enc_lin(ev_keys)).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "pred_Linear_Ridge.npy"), pred_ridge)

    # 3. Static-MLP Baseline (ESM2 + Coexp + Ctrl PCA-free mean, no OmniPath/Pathway priors)
    ctrl_tr = np.vstack([lineage_ctrls[ln] for ln in tr_lineages]).astype(np.float32)
    ctrl_ev = np.repeat(mu_rpe1[None, :], len(ev_keys), axis=0).astype(np.float32)
    X_mlp_tr = np.hstack([X_esm_tr, X_coexp_tr, ctrl_tr[:, :128]]).astype(np.float32)
    X_mlp_ev = np.hstack([X_esm_ev, X_coexp_ev, ctrl_ev[:, :128]]).astype(np.float32)

    torch.manual_seed(SEED)
    mlp = StaticMLPPert(X_mlp_tr.shape[1], n_genes, hid=256).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_mlp_tr), torch.tensor(Y_tr)), batch_size=256, shuffle=True)
    mlp.train()
    for _ in range(30):
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(mlp(bx), by)
            loss.backward()
            opt.step()
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(torch.tensor(X_mlp_ev, device=DEVICE)).cpu().numpy().astype(np.float32)
    np.save(os.path.join(OUT_DIR, "pred_Static_MLP.npy"), pred_mlp)

    for mname, pmat in [("Train_Mean", pred_mean), ("Linear_Ridge", pred_ridge), ("Static_MLP", pred_mlp)]:
        m = evaluate_perturbation_predictions(d_rpe1, {k: pmat[i] for i, k in enumerate(ev_keys)}, ev_keys)
        print(f"  [{mname:14s}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
    print(f"Part 1 completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
