# -*- coding: utf-8 -*-
"""
exp37b_overexpression_ridge_anchored.py
Exp37b: Optimized Gene Overexpression World Model:
  1. Fast chunked streaming of JoungZhang2023_atlas.h5ad (saves reusable cache).
  2. Upgrades OverexpressionOmniPathNet with Exp31-style LOO Ridge-Anchored Residual Gate
     + 5-Member Deep Ensemble so small-sample CRISPRa (84 train genes) never underfits
     while retaining the +0.0803 JoungZhang2023 ORF pretraining & OmniPath causal gain.
  3. Evaluates on the exact same Norman2019 Unseen-Gene CRISPRa OOD split (84 -> 21) as Exp37.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp3_triteacher_distill import load_esm2_gene_embeddings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
JOUNG_CACHE = os.path.join(DATA_DIR, "joung_orf_norman_hvg2000_cache.npz")
OUT_CSV = r"D:\data\AIVC\test_results\exp37b_overexpression_optimized_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp37b_overexpression_optimized_plot.png"


class RidgeAnchoredOverexprNet(nn.Module):
    """
    Exp31-style Ridge-Anchored + OmniPath Causal Overexpression World Model:
      pred = gate(x_stat) * (ridge_scale * x_ridge) + res_scale * (head(esm, causal) @ U_prog)
    """

    def __init__(self, d_esm: int, k_prog: int, U_prog: np.ndarray, stat_dim: int = 3, hid: int = 128):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.res_scale = nn.Parameter(torch.tensor(0.35))

        self.target_gate = nn.Sequential(nn.Linear(stat_dim, 24), nn.GELU(), nn.Linear(24, 1))
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)

        self.esm_enc = nn.Sequential(nn.Linear(d_esm, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.25))
        self.causal_enc = nn.Sequential(nn.Linear(k_prog, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.25))
        self.head = nn.Sequential(
            nn.Linear(hid + 48, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward_residual(self, x_esm: torch.Tensor, x_causal: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.esm_enc(x_esm), self.causal_enc(x_causal)], dim=-1)
        return self.head(h) @ self.U_prog

    def forward(self, x_esm: torch.Tensor, x_causal: torch.Tensor, x_ridge: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        gate = 1.0 + 0.35 * torch.tanh(self.target_gate(x_stat))
        return gate * (self.ridge_scale * x_ridge) + self.res_scale * self.forward_residual(x_esm, x_causal)


def extract_joung_orf_fast(target_genes: list, max_orfs: int = 500, max_cells_per_orf: int = 120):
    if os.path.exists(JOUNG_CACHE):
        ck = np.load(JOUNG_CACHE, allow_pickle=True)
        print(f"  [Joung Cache] Loaded {len(ck['orf_genes'])} ORFs from {JOUNG_CACHE}")
        return [str(g) for g in ck["orf_genes"]], ck["orf_deltas"].astype(np.float32)

    t_s = time.time()
    fp = os.path.join(DATA_DIR, "JoungZhang2023_atlas.h5ad")
    a = ad.read_h5ad(fp, backed="r")
    j_genes = [str(g) for g in a.var_names]
    j_g2i = {g.upper(): i for i, g in enumerate(j_genes)}

    shared_t_idx, shared_j_idx = [], []
    for ti, g in enumerate(target_genes):
        ji = j_g2i.get(g.upper())
        if ji is not None:
            shared_t_idx.append(ti)
            shared_j_idx.append(ji)
    shared_t_idx = np.array(shared_t_idx, dtype=np.int64)
    shared_j_idx = np.array(shared_j_idx, dtype=np.int64)

    obs_pert = a.obs["perturbation"].astype(str).values
    vc = pd.Series(obs_pert).value_counts()
    cand_orfs = [str(k) for k in vc.index if str(k) not in ("3368", "3367") and vc[k] >= 60][:max_orfs]
    wanted_set = set(cand_orfs) | {"3368"}

    rng = np.random.RandomState(SEED)
    selected_rows = {}
    ctrl_pos = np.where(obs_pert == "3368")[0]
    selected_rows["3368"] = rng.choice(ctrl_pos, size=min(4000, len(ctrl_pos)), replace=False)
    for orf_id in cand_orfs:
        pos = np.where(obs_pert == orf_id)[0]
        selected_rows[orf_id] = rng.choice(pos, size=min(max_cells_per_orf, len(pos)), replace=False)

    # Map each selected row index -> group key, then read sequentially in contiguous chunks!
    row_to_grp = {}
    for k, idxs in selected_rows.items():
        for idx in idxs:
            row_to_grp[int(idx)] = k
    all_rows = sorted(row_to_grp.keys())
    print(f"  [Joung Fast Stream] Reading {len(all_rows)} cells across {len(cand_orfs)} ORFs in chunked sequential order...")

    sum_vecs = {k: np.zeros(len(j_genes), dtype=np.float64) for k in wanted_set}
    cnt_vecs = {k: 0 for k in wanted_set}

    chunk_size = 5000
    for s0 in range(0, len(all_rows), chunk_size):
        sub_r = all_rows[s0:s0 + chunk_size]
        # Read contiguous slice from min(sub_r) to max(sub_r)+1 if dense enough, else fancy index
        X_sub = a.X[sub_r, :]
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        X_sub = np.log1p(np.clip(np.asarray(X_sub, dtype=np.float32), 0.0, None))
        for local_i, r_idx in enumerate(sub_r):
            gk = row_to_grp[r_idx]
            sum_vecs[gk] += X_sub[local_i]
            cnt_vecs[gk] += 1
    a.file.close()

    mu_ctrl_j = (sum_vecs["3368"] / max(cnt_vecs["3368"], 1)).astype(np.float32)
    mu_ctrl_shared = mu_ctrl_j[shared_j_idx]

    orf_genes, orf_deltas = [], []
    for orf_id in cand_orfs:
        if cnt_vecs[orf_id] < 20:
            continue
        mu_p = (sum_vecs[orf_id] / cnt_vecs[orf_id]).astype(np.float32)
        full_delta = mu_p - mu_ctrl_j
        top_j = int(np.argmax(full_delta))
        if full_delta[top_j] < 0.25:
            continue
        t_gene = j_genes[top_j].upper()
        dv_target = np.zeros(len(target_genes), dtype=np.float32)
        dv_target[shared_t_idx] = (mu_p[shared_j_idx] - mu_ctrl_shared).astype(np.float32)
        orf_genes.append(t_gene)
        orf_deltas.append(dv_target)

    mat = np.vstack(orf_deltas).astype(np.float32)
    np.savez_compressed(JOUNG_CACHE, orf_genes=np.array(orf_genes), orf_deltas=mat)
    print(f"  [Joung Cache] Extracted & cached {len(mat)} ORFs in {time.time()-t_s:.1f}s -> {JOUNG_CACHE}")
    return orf_genes, mat


def main():
    t0 = time.time()
    print(f"=== Exp37b: Ridge-Anchored Overexpression World Model (JoungZhang2023 -> Norman2019) ({DEVICE}) ===")

    norm_ad = sc.read_h5ad(os.path.join(DATA_DIR, "Norman2019_hvg2000.h5ad"))
    target_genes = [str(g) for g in norm_ad.var_names]
    n_genes = len(target_genes)
    gene_to_idx = {g.upper(): i for i, g in enumerate(target_genes)}

    mu_ctrl = _to_dense_array(norm_ad[norm_ad.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_norm = _compute_group_deltas(norm_ad[norm_ad.obs["perturbation"] != "control"], "perturbation", mu_ctrl)
    del norm_ad

    single_perts = sorted([p for p in d_norm.keys() if "_" not in p and "+" not in p])
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(single_perts)
    n_tr = int(0.8 * len(single_perts))
    tr_perts = sorted(perm[:n_tr].tolist())
    ev_perts = sorted(perm[n_tr:].tolist())

    Y_tr = np.vstack([d_norm[p] for p in tr_perts]).astype(np.float32)
    d_ev = {p: d_norm[p] for p in ev_perts}

    ev_set_up = {p.upper() for p in ev_perts}
    j_genes, Y_joung_raw = extract_joung_orf_fast(target_genes, max_orfs=500, max_cells_per_orf=120)
    keep_j = [i for i, g in enumerate(j_genes) if g.upper() not in ev_set_up]
    j_genes = [j_genes[i] for i in keep_j]
    Y_joung = Y_joung_raw[keep_j]
    Y_joung = Y_joung / (np.std(Y_joung) + 1e-6) * float(np.std(Y_tr))

    esm2_dict = load_esm2_gene_embeddings(os.path.join(DATA_DIR, "target_esm2_150m.npz"))
    esm2_up = {str(k).upper(): v for k, v in esm2_dict.items()}
    d_esm = len(next(iter(esm2_up.values())))

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

    svd = TruncatedSVD(n_components=48, random_state=SEED).fit(np.vstack([Y_tr, Y_joung]))
    U_prog = svd.components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

    def build_feats(plist):
        xes, xca, xst = [], [], []
        for p in plist:
            pu = str(p).upper()
            xes.append(esm2_up.get(pu, np.zeros(d_esm, dtype=np.float32)))
            gi = gene_to_idx.get(pu, -1)
            if gi >= 0:
                xca.append(causal_gene_mat[gi])
                raw_e = float(mu_ctrl[gi])
                z_e = float((mu_ctrl[gi] - mu_ctrl.mean()) / (mu_ctrl.std() + 1e-6))
            else:
                xca.append(np.zeros(48, dtype=np.float32))
                raw_e, z_e = 0.0, 0.0
            xst.append([raw_e, z_e, 1.0 if gi >= 0 else 0.0])
        return np.asarray(xes, dtype=np.float32), np.asarray(xca, dtype=np.float32), np.asarray(xst, dtype=np.float32)

    X_esm_tr, X_cau_tr, X_st_tr = build_feats(tr_perts)
    X_esm_ev, X_cau_ev, X_st_ev = build_feats(ev_perts)
    X_esm_j, X_cau_j, _ = build_feats(j_genes)

    # 1. Out-of-fold Ridge predictions on Train (to prevent leakage) + Full Ridge on Test
    ridge_oof_tr = np.zeros_like(Y_tr)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_i, va_i in kf.split(X_esm_tr):
        r_cv = Ridge(alpha=50.0).fit(X_esm_tr[tr_i], Y_tr[tr_i])
        ridge_oof_tr[va_i] = r_cv.predict(X_esm_tr[va_i]).astype(np.float32)
    ridge_full = Ridge(alpha=50.0).fit(X_esm_tr, Y_tr)
    p_ridge_ev = ridge_full.predict(X_esm_ev).astype(np.float32)

    # Joung residual target (centered by Joung mean so pretraining learns differential activation!)
    Y_joung_res = (Y_joung - Y_joung.mean(axis=0, keepdims=True)).astype(np.float32)

    # Baselines
    p_mean = np.repeat(Y_tr.mean(axis=0, keepdims=True), len(ev_perts), axis=0).astype(np.float32)
    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(d_esm, 160), nn.LayerNorm(160), nn.GELU(), nn.Dropout(0.25), nn.Linear(160, n_genes)).to(DEVICE)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=2e-3)
    for _ in range(80):
        opt_m.zero_grad()
        F.mse_loss(mlp(torch.tensor(X_esm_tr, device=DEVICE)), torch.tensor(Y_tr, device=DEVICE)).backward()
        opt_m.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp = mlp(torch.tensor(X_esm_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

    # 5-Member Deep Ensemble of Ridge-Anchored OverexpressionOmniPathNet (Joung Pretrained)
    ens_preds = []
    for sd_val in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd_val)
        net = RidgeAnchoredOverexprNet(d_esm, 48, U_prog, stat_dim=3, hid=128).to(DEVICE)
        # Stage 1: Pretrain residual pathway branch on JoungZhang2023 ORF overexpression
        opt_p = torch.optim.AdamW(
            list(net.esm_enc.parameters()) + list(net.causal_enc.parameters()) + list(net.head.parameters()),
            lr=1.5e-3, weight_decay=2e-3
        )
        dl_p = DataLoader(TensorDataset(torch.tensor(X_esm_j), torch.tensor(X_cau_j), torch.tensor(Y_joung_res)), batch_size=64, shuffle=True)
        net.train()
        for _ in range(30):
            for bx, bc, by in dl_p:
                bx, bc, by = bx.to(DEVICE), bc.to(DEVICE), by.to(DEVICE)
                opt_p.zero_grad()
                F.mse_loss(net.forward_residual(bx, bc), by).backward()
                opt_p.step()

        # Stage 2: Fine-tune full Ridge-Anchored network on Norman2019
        opt_f = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=3e-3)
        xe_t = torch.tensor(X_esm_tr, device=DEVICE)
        xc_t = torch.tensor(X_cau_tr, device=DEVICE)
        xr_t = torch.tensor(ridge_oof_tr, device=DEVICE)
        xs_t = torch.tensor(X_st_tr, device=DEVICE)
        y_t = torch.tensor(Y_tr, device=DEVICE)
        for _ in range(60):
            opt_f.zero_grad()
            F.mse_loss(net(xe_t, xc_t, xr_t, xs_t), y_t).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt_f.step()
        net.eval()
        with torch.no_grad():
            pev = net(
                torch.tensor(X_esm_ev, device=DEVICE),
                torch.tensor(X_cau_ev, device=DEVICE),
                torch.tensor(p_ridge_ev, device=DEVICE),
                torch.tensor(X_st_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        ens_preds.append(pev)

    preds_dict = {
        "Train_Mean": p_mean,
        "Linear_Ridge_ESM2": p_ridge_ev,
        "Static_MLP_ESM2": p_mlp,
        "AIVC_Overexpr_RidgeAnchored_Single": ens_preds[0],
        "AIVC_Overexpr_RidgeAnchored_Ens5": np.mean(ens_preds, axis=0).astype(np.float32),
    }

    rows = []
    for mname, pmat in preds_dict.items():
        df_m = evaluate_perturbation_predictions(d_ev, {ev_perts[i]: pmat[i] for i in range(len(ev_perts))}, model_name=mname, top_k_list=[20])
        row = {
            "model": mname,
            "mean_pearson": float(df_m["pearson_delta_all"].mean()),
            "top20_pearson": float(df_m["pearson_delta_top20"].mean()),
            "dir_acc_top20": float(df_m["dir_acc_top20"].mean()),
            "mse": float(df_m["mse_all"].mean()),
        }
        rows.append(row)
        print(f"  [{mname:34s}] MeanR={row['mean_pearson']:.4f} | Top20R={row['top20_pearson']:.4f} | DirAcc={row['dir_acc_top20']:.4f} | MSE={row['mse']:.6f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(res_df))
    w = 0.26
    ax.bar(x - w, res_df["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, res_df["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, res_df["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in res_df["model"]], fontsize=8)
    ax.set_title("Exp37b: Ridge-Anchored Gene Overexpression World Model\n(JoungZhang2023 ORF Pretrained -> Norman2019 Unseen CRISPRa)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp37b results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
