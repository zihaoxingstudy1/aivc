# -*- coding: utf-8 -*-
"""
exp37c_overexpression_cis_omnipath.py
Exp37c (10s using cached 485 JoungZhang2023 ORFs):
  Adds explicit Cis-Activation + OmniPath 1-Hop Directed Downstream Prior
  to Ridge-Anchored Overexpression World Model on Norman2019 Unseen CRISPRa (84 -> 21).
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
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
OUT_CSV = r"D:\data\AIVC\test_results\exp37c_overexpression_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp37c_overexpression_plot.png"


class CisOmniPathOverexprModel(nn.Module):
    """
    Mechanistic Overexpression Model:
      pred = ridge_scale * pred_ridge
           + alpha_cis * cis_onehot
           + alpha_op1 * omn_1hop
           + alpha_op2 * omn_2hop
           + 0.15 * (low_rank_head(esm, causal) @ U_prog)
    """

    def __init__(self, d_esm: int, k_prog: int, U_prog: np.ndarray, hid: int = 96):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.alpha_cis = nn.Parameter(torch.tensor(0.2))
        self.alpha_op1 = nn.Parameter(torch.tensor(0.05))
        self.alpha_op2 = nn.Parameter(torch.tensor(0.02))
        self.net = nn.Sequential(
            nn.Linear(d_esm + k_prog, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward_mech(self, x_esm, x_causal, x_cis, x_op1, x_op2):
        delta_lr = self.net(torch.cat([x_esm, x_causal], dim=-1)) @ self.U_prog
        return self.alpha_cis * x_cis + self.alpha_op1 * x_op1 + self.alpha_op2 * x_op2 + 0.15 * delta_lr

    def forward(self, x_esm, x_causal, x_ridge, x_cis, x_op1, x_op2):
        return self.ridge_scale * x_ridge + self.forward_mech(x_esm, x_causal, x_cis, x_op1, x_op2)


def main():
    t0 = time.time()
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

    ck = np.load(JOUNG_CACHE, allow_pickle=True)
    j_genes_all, Y_joung_all = [str(g) for g in ck["orf_genes"]], ck["orf_deltas"].astype(np.float32)
    ev_set_up = {p.upper() for p in ev_perts}
    keep_j = [i for i, g in enumerate(j_genes_all) if g.upper() not in ev_set_up]
    j_genes = [j_genes_all[i] for i in keep_j]
    Y_joung = Y_joung_all[keep_j]
    Y_joung = Y_joung / (np.std(Y_joung) + 1e-6) * float(np.std(Y_tr))
    Y_joung_res = (Y_joung - Y_joung.mean(axis=0, keepdims=True)).astype(np.float32)

    esm2_dict = load_esm2_gene_embeddings(os.path.join(DATA_DIR, "target_esm2_150m.npz"))
    esm2_up = {str(k).upper(): v for k, v in esm2_dict.items()}
    d_esm = len(next(iter(esm2_up.values())))

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A2 = A_signed @ A_signed

    svd = TruncatedSVD(n_components=32, random_state=SEED).fit(np.vstack([Y_tr, Y_joung]))
    U_prog = svd.components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * A2) @ U_prog.T).astype(np.float32)

    def build_feats(plist):
        xes, xca, xcis, xop1, xop2 = [], [], [], [], []
        for p in plist:
            pu = str(p).upper()
            xes.append(esm2_up.get(pu, np.zeros(d_esm, dtype=np.float32)))
            gi = gene_to_idx.get(pu, -1)
            c_oh = np.zeros(n_genes, dtype=np.float32)
            if gi >= 0:
                c_oh[gi] = 1.0
                xca.append(causal_gene_mat[gi])
                xop1.append(A_signed[:, gi])
                xop2.append(A2[:, gi])
            else:
                xca.append(np.zeros(32, dtype=np.float32))
                xop1.append(np.zeros(n_genes, dtype=np.float32))
                xop2.append(np.zeros(n_genes, dtype=np.float32))
            xcis.append(c_oh)
        return (
            np.asarray(xes, dtype=np.float32),
            np.asarray(xca, dtype=np.float32),
            np.asarray(xcis, dtype=np.float32),
            np.asarray(xop1, dtype=np.float32),
            np.asarray(xop2, dtype=np.float32),
        )

    X_esm_tr, X_ca_tr, X_cis_tr, X_op1_tr, X_op2_tr = build_feats(tr_perts)
    X_esm_ev, X_ca_ev, X_cis_ev, X_op1_ev, X_op2_ev = build_feats(ev_perts)
    X_esm_j, X_ca_j, X_cis_j, X_op1_j, X_op2_j = build_feats(j_genes)

    ridge_oof_tr = np.zeros_like(Y_tr)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_i, va_i in kf.split(X_esm_tr):
        r_cv = Ridge(alpha=50.0).fit(X_esm_tr[tr_i], Y_tr[tr_i])
        ridge_oof_tr[va_i] = r_cv.predict(X_esm_tr[va_i]).astype(np.float32)
    ridge_full = Ridge(alpha=50.0).fit(X_esm_tr, Y_tr)
    p_ridge_ev = ridge_full.predict(X_esm_ev).astype(np.float32)

    ens_preds = []
    for sd_val in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd_val)
        net = CisOmniPathOverexprModel(d_esm, 32, U_prog, hid=96).to(DEVICE)
        # Stage 1: Pretrain on JoungZhang2023 ORF overexpression residuals
        opt_p = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-3)
        xej = torch.tensor(X_esm_j, device=DEVICE)
        xcj = torch.tensor(X_ca_j, device=DEVICE)
        xij = torch.tensor(X_cis_j, device=DEVICE)
        x1j = torch.tensor(X_op1_j, device=DEVICE)
        x2j = torch.tensor(X_op2_j, device=DEVICE)
        yj = torch.tensor(Y_joung_res, device=DEVICE)
        net.train()
        for _ in range(35):
            opt_p.zero_grad()
            F.mse_loss(net.forward_mech(xej, xcj, xij, x1j, x2j), yj).backward()
            opt_p.step()

        # Stage 2: Fine-tune on Norman2019 (Ridge-anchored)
        opt_f = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=5e-3)
        xet, xct, xrt = torch.tensor(X_esm_tr, device=DEVICE), torch.tensor(X_ca_tr, device=DEVICE), torch.tensor(ridge_oof_tr, device=DEVICE)
        xit, x1t, x2t, yt = torch.tensor(X_cis_tr, device=DEVICE), torch.tensor(X_op1_tr, device=DEVICE), torch.tensor(X_op2_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
        for _ in range(45):
            opt_f.zero_grad()
            F.mse_loss(net(xet, xct, xrt, xit, x1t, x2t), yt).backward()
            opt_f.step()
        net.eval()
        with torch.no_grad():
            pev = net(
                torch.tensor(X_esm_ev, device=DEVICE),
                torch.tensor(X_ca_ev, device=DEVICE),
                torch.tensor(p_ridge_ev, device=DEVICE),
                torch.tensor(X_cis_ev, device=DEVICE),
                torch.tensor(X_op1_ev, device=DEVICE),
                torch.tensor(X_op2_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        ens_preds.append(pev)
        if sd_val == 42:
            print(f"  Learned priors (seed=42): ridge_scale={net.ridge_scale.item():.3f}, alpha_cis={net.alpha_cis.item():.3f}, alpha_op1={net.alpha_op1.item():.3f}, alpha_op2={net.alpha_op2.item():.3f}")

    p_mean = np.repeat(Y_tr.mean(axis=0, keepdims=True), len(ev_perts), axis=0).astype(np.float32)
    preds_dict = {
        "Train_Mean": p_mean,
        "Linear_Ridge_ESM2": p_ridge_ev,
        "AIVC_Overexpr_CisOmniPath_Single": ens_preds[0],
        "AIVC_Overexpr_CisOmniPath_Ens5": np.mean(ens_preds, axis=0).astype(np.float32),
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

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
    x = np.arange(len(res_df))
    w = 0.26
    ax.bar(x - w, res_df["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, res_df["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, res_df["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in res_df["model"]], fontsize=8.5)
    ax.set_title("Exp37c: Cis-Activation + OmniPath + JoungZhang2023 Pretrained Overexpression", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp37c results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
