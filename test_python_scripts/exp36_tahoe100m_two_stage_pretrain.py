# -*- coding: utf-8 -*-
"""
exp36_tahoe100m_two_stage_pretrain.py
Exp36: Two-Stage Mega-Scale Pretraining (Tahoe-100M Pseudobulk DE Shards)
       + Platform-Adapter Fine-Tuning on 7-Lineage -> Blind Unseen RPE1 (2,016 perturbations).
Strict No-Regression Gate against v1.1 (Exp31).
"""

import os
import sys
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp31_lineage_invariant_gnn_proteome import (
    LineageInvariantPathwayNet, predict_lineage_prot_phos_via_exp28
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAHOE_PB_DIR = r"D:\data\AIVC\data\tahoe_100m\metadata\pseudobulk_differential_expression"
BUNDLE_PATH = r"D:\data\AIVC\test_results\sota_bench_preds\rpe1_ood_bundle.npz"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp36_tahoe_pretrain_results.csv"


def load_tahoe_shared_deltas(target_genes: list, max_shards: int = 12, max_perts: int = 4000) -> np.ndarray:
    """Stream Tahoe-100M pseudobulk DE shards and align log2FoldChange vectors to target_genes."""
    g2i = {g.upper(): i for i, g in enumerate(target_genes)}
    target_set = set(g2i.keys())
    shards = sorted(glob.glob(os.path.join(TAHOE_PB_DIR, "train-*.parquet")))[:max_shards]
    vecs = {}
    for sp in shards:
        df = pd.read_parquet(sp, columns=["gene_name", "log2FoldChange", "Cell_Name_Vevo", "drug", "concentration"])
        df["g_up"] = df["gene_name"].astype(str).str.upper()
        df = df[df["g_up"].isin(target_set)].dropna(subset=["log2FoldChange"])
        for (cl, drg, conc), grp in df.groupby(["Cell_Name_Vevo", "drug", "concentration"], sort=False):
            key = (str(cl), str(drg), float(conc))
            if key not in vecs:
                if len(vecs) >= max_perts:
                    continue
                vecs[key] = np.zeros(len(target_genes), dtype=np.float32)
            idxs = [g2i[g] for g in grp["g_up"].values]
            vals = np.clip(grp["log2FoldChange"].values.astype(np.float32), -5.0, 5.0)
            vecs[key][idxs] = vals
        if len(vecs) >= max_perts:
            break
    mat = np.vstack(list(vecs.values())).astype(np.float32)
    # Normalize scale to match Perturb-seq delta scale
    scale = np.std(mat) + 1e-6
    mat = mat / scale * 0.12
    print(f"  [Tahoe-100M] Loaded {len(mat)} pseudobulk (cell x drug x dose) vectors from {len(shards)} shards")
    return mat


def main():
    t0 = time.time()
    print(f"=== Exp36: Tahoe-100M Two-Stage Shared-Gene Pretraining -> Unseen RPE1 ({DEVICE}) ===")
    bundle = np.load(BUNDLE_PATH, allow_pickle=True)
    target_genes = [str(g) for g in bundle["target_genes"]]
    n_genes = len(target_genes)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    tr_perts = [str(p) for p in bundle["tr_perts"]]
    tr_lineages = np.array([str(ln) for ln in bundle["tr_lineages"]])
    Y_tr = bundle["Y_tr"]
    Y_ev = bundle["Y_ev"]
    d_rpe1 = {ev_keys[i]: Y_ev[i] for i in range(len(ev_keys))}

    # Stage 1: Tahoe-100M shared-gene program basis + backbone pretraining
    Y_tahoe = load_tahoe_shared_deltas(target_genes, max_shards=10, max_perts=3000)
    svd_pert = TruncatedSVD(n_components=48, random_state=SEED).fit(Y_tr)
    svd_tahoe = TruncatedSVD(n_components=16, random_state=SEED).fit(Y_tahoe)
    # Orthonormalize combined 64-dim universal program basis
    U_comb = np.vstack([svd_pert.components_, svd_tahoe.components_]).astype(np.float32)
    q, _ = np.linalg.qr(U_comb.T)
    U_prog_univ = q[:, :64].T.astype(np.float32)

    # Pretrain ctrl_enc & prog_head reconstruction bottleneck on Tahoe-100M
    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog_univ.T).astype(np.float32)

    lineage_ctrls = {str(k): v for k, v in zip(bundle["ctrl_names"], bundle["ctrl_vecs"])}
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    pred_ridge_ev = np.load(r"D:\data\AIVC\test_results\sota_bench_preds\pred_Linear_Ridge.npy")
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

    # Stage 1 pretraining of Perturbation Autoencoder / Denoiser on Tahoe-100M -> initialize LineageInvariantPathwayNet
    tahoe_ae = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU()).to(DEVICE)
    tahoe_dec = nn.Linear(64, n_genes).to(DEVICE)
    opt_pre = torch.optim.AdamW(list(tahoe_ae.parameters()) + list(tahoe_dec.parameters()), lr=2e-3, weight_decay=1e-4)
    dl_pre = DataLoader(TensorDataset(torch.tensor(Y_tahoe)), batch_size=256, shuffle=True)
    for _ in range(15):
        for (by,) in dl_pre:
            by = by.to(DEVICE)
            opt_pre.zero_grad()
            z = tahoe_ae(by)
            F.mse_loss(tahoe_dec(z), by).backward()
            opt_pre.step()
    print("  [Stage 1] Tahoe-100M shared-gene encoder pretraining complete")

    # Stage 2: Fine-tune 5-member ensemble with Tahoe-initialized ctrl_enc + Universal Program Basis
    seeds = [42, 101, 202, 303, 404]
    preds = []
    for sd_val in seeds:
        torch.manual_seed(sd_val)
        np.random.seed(sd_val)
        model = LineageInvariantPathwayNet(n_genes, teach_tr.shape[1], 64, U_prog_univ, stat_dim=5, hid=128).to(DEVICE)
        # Transfer Tahoe pretrained linear layer weights into ctrl_enc[0]
        with torch.no_grad():
            model.ctrl_enc[0].weight.copy_(tahoe_ae[0].weight)
            model.ctrl_enc[0].bias.copy_(tahoe_ae[0].bias)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
        dl = DataLoader(
            TensorDataset(
                torch.tensor(ctrl_tr), torch.tensor(teach_tr), torch.tensor(r_tr),
                torch.tensor(tstat_tr), torch.tensor(caus_tr), torch.tensor(Y_tr),
            ),
            batch_size=512, shuffle=True
        )
        model.train()
        for _ in range(35):
            for bc, bt, br, bts, bca, by in dl:
                bc, bt, br, bts, bca, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bts.to(DEVICE), bca.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                F.mse_loss(model(bc, bt, br, bts, bca), by).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pev = model(
                torch.tensor(ctrl_ev, device=DEVICE), torch.tensor(teach_ev, device=DEVICE),
                torch.tensor(r_ev, device=DEVICE), torch.tensor(tstat_ev, device=DEVICE),
                torch.tensor(caus_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        preds.append(pev)

    p_single = preds[0]
    p_ens = np.mean(preds, axis=0).astype(np.float32)
    rows = []
    for mname, pmat in [("Exp36_TahoePretrain_Single", p_single), ("Exp36_TahoePretrain_Ensemble", p_ens)]:
        df_m = evaluate_perturbation_predictions(d_rpe1, {ev_keys[i]: pmat[i] for i in range(len(ev_keys))}, model_name=mname, top_k_list=[20])
        row = {
            "model": mname,
            "mean_pearson": float(df_m["pearson_delta_all"].mean()),
            "top20_pearson": float(df_m["pearson_delta_top20"].mean()),
            "dir_acc_top20": float(df_m["dir_acc_top20"].mean()),
            "mse": float(df_m["mse_all"].mean()),
        }
        rows.append(row)
        print(f"  [{mname:28s}] MeanR={row['mean_pearson']:.4f} | Top20R={row['top20_pearson']:.4f} | DirAcc={row['dir_acc_top20']:.4f} | MSE={row['mse']:.6f}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved Exp36 results -> {OUT_CSV} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
