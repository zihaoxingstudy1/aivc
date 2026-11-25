# -*- coding: utf-8 -*-
"""
exp39_norman_combo_epistasis_benchmark.py
Exp39: Norman2019 Unseen Double-Gene Combinatorial Overexpression & Epistasis (Synergy) Benchmark
  - Compares Train-Mean, Additive (dA + dB), Linear-Ridge, Static-MLP, and
    AIVC_Combo_Epistasis_Ens5 (Additive Anchor + OmniPath Cross-Talk Interaction + ESM2 Bilinear).
  - Evaluates both overall Combo accuracy (MeanR, Top20R, DirAcc, MSE) and
    Non-Additive Epistasis residual correlation (true_delta - (dA + dB)).
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
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
OUT_CSV = r"D:\data\AIVC\test_results\exp39_norman_combo_epistasis_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp39_norman_combo_epistasis_plot.png"


class ComboEpistasisOmniPathNet(nn.Module):
    """
    Predicts double-gene combinatorial perturbation delta:
      pred = w_add * (dA + dB) + w_dom * max_abs(dA, dB) + w_syn * (epistasis_head(feats) @ U_prog)
             + alpha_xtalk * (opA * opB)
    """

    def __init__(self, d_in: int, k_prog: int, U_prog: np.ndarray, hid: int = 128):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.w_add = nn.Parameter(torch.tensor(0.80))
        self.w_dom = nn.Parameter(torch.tensor(0.15))
        self.alpha_xtalk = nn.Parameter(torch.tensor(0.05))
        self.epistasis_head = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.epistasis_head[-1].weight)
        nn.init.zeros_(self.epistasis_head[-1].bias)

    def forward(self, x_add, x_dom, x_feat, x_xtalk):
        syn = self.epistasis_head(x_feat) @ self.U_prog
        return self.w_add * x_add + self.w_dom * x_dom + self.alpha_xtalk * x_xtalk + 0.35 * syn


def eval_epistasis(y_true: np.ndarray, y_pred: np.ndarray, y_add: np.ndarray) -> tuple:
    true_epi = y_true - y_add
    pred_epi = y_pred - y_add
    rs = []
    for i in range(len(y_true)):
        if np.std(true_epi[i]) > 1e-6 and np.std(pred_epi[i]) > 1e-6:
            r = pearsonr(true_epi[i], pred_epi[i])[0]
            if np.isfinite(r):
                rs.append(r)
    epi_r = float(np.mean(rs)) if rs else 0.0
    epi_mse = float(np.mean((true_epi - pred_epi) ** 2))
    return epi_r, epi_mse


def main():
    t0 = time.time()
    print(f"=== Exp39: Norman2019 Unseen Double-Gene Combo & Epistasis Benchmark ({DEVICE}) ===")

    norm_ad = sc.read_h5ad(os.path.join(DATA_DIR, "Norman2019_hvg2000.h5ad"))
    target_genes = [str(g) for g in norm_ad.var_names]
    n_genes = len(target_genes)
    gene_to_idx = {g.upper(): i for i, g in enumerate(target_genes)}

    mu_ctrl = _to_dense_array(norm_ad[norm_ad.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_norm = _compute_group_deltas(norm_ad[norm_ad.obs["perturbation"] != "control"], "perturbation", mu_ctrl)
    del norm_ad

    singles = {p: v for p, v in d_norm.items() if "_" not in p and "+" not in p}
    combos_valid = []
    for p, v in d_norm.items():
        if "_" in p:
            parts = p.split("_")
            if len(parts) == 2 and parts[0] in singles and parts[1] in singles:
                combos_valid.append((p, parts[0], parts[1], v))
    combos_valid.sort(key=lambda x: x[0])

    rng = np.random.RandomState(SEED)
    idx_perm = rng.permutation(len(combos_valid))
    n_tr = int(0.8 * len(combos_valid))
    tr_combos = [combos_valid[i] for i in idx_perm[:n_tr]]
    ev_combos = [combos_valid[i] for i in idx_perm[n_tr:]]
    print(f"  Singles={len(singles)} | Valid Paired Combos={len(combos_valid)} -> Train={len(tr_combos)}, Unseen Combo Test={len(ev_combos)}")

    esm2_dict = load_esm2_gene_embeddings(os.path.join(DATA_DIR, "target_esm2_150m.npz"))
    esm2_up = {str(k).upper(): v for k, v in esm2_dict.items()}
    d_esm = len(next(iter(esm2_up.values())))

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

    Y_tr = np.vstack([c[3] for c in tr_combos]).astype(np.float32)
    Y_ev = np.vstack([c[3] for c in ev_combos]).astype(np.float32)
    ev_keys = [c[0] for c in ev_combos]
    d_ev = {c[0]: c[3] for c in ev_combos}

    svd = TruncatedSVD(n_components=32, random_state=SEED).fit(np.vstack(list(singles.values()) + [Y_tr]))
    U_prog = svd.components_.astype(np.float32)

    def build_combo_tensors(clist):
        x_add_l, x_dom_l, x_feat_l, x_xt_l = [], [], [], []
        for _, gA, gB, _ in clist:
            dA, dB = singles[gA], singles[gB]
            x_add_l.append(dA + dB)
            # Dominant-effect gene-wise envelope
            x_dom_l.append(np.where(np.abs(dA) >= np.abs(dB), dA, dB))
            eA = esm2_up.get(gA.upper(), np.zeros(d_esm, dtype=np.float32))
            eB = esm2_up.get(gB.upper(), np.zeros(d_esm, dtype=np.float32))
            zA = dA @ U_prog.T
            zB = dB @ U_prog.T
            # Symmetric interaction features: (zA+zB, zA*zB, eA+eB, eA*eB)
            x_feat_l.append(np.concatenate([zA + zB, zA * zB, eA + eB, eA * eB]))
            giA = gene_to_idx.get(gA.upper(), -1)
            giB = gene_to_idx.get(gB.upper(), -1)
            opA = A_signed[:, giA] if giA >= 0 else np.zeros(n_genes, dtype=np.float32)
            opB = A_signed[:, giB] if giB >= 0 else np.zeros(n_genes, dtype=np.float32)
            x_xt_l.append(opA * opB)
        return (
            np.asarray(x_add_l, dtype=np.float32),
            np.asarray(x_dom_l, dtype=np.float32),
            np.asarray(x_feat_l, dtype=np.float32),
            np.asarray(x_xt_l, dtype=np.float32),
        )

    Xadd_tr, Xdom_tr, Xf_tr, Xxt_tr = build_combo_tensors(tr_combos)
    Xadd_ev, Xdom_ev, Xf_ev, Xxt_ev = build_combo_tensors(ev_combos)

    # 1. Train-Mean
    p_mean = np.repeat(Y_tr.mean(axis=0, keepdims=True), len(ev_combos), axis=0).astype(np.float32)
    # 2. Additive Oracle (dA + dB)
    p_add = Xadd_ev.copy()
    # 3. Linear-Ridge on Symmetric Features
    ridge = Ridge(alpha=50.0).fit(Xf_tr, Y_tr)
    p_ridge = ridge.predict(Xf_ev).astype(np.float32)
    # 4. Static-MLP on Symmetric Features (no additive/OmniPath anchor)
    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(Xf_tr.shape[1], 160), nn.LayerNorm(160), nn.GELU(), nn.Dropout(0.3), nn.Linear(160, n_genes)).to(DEVICE)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=1.5e-3, weight_decay=3e-3)
    for _ in range(90):
        opt_m.zero_grad()
        F.mse_loss(mlp(torch.tensor(Xf_tr, device=DEVICE)), torch.tensor(Y_tr, device=DEVICE)).backward()
        opt_m.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp = mlp(torch.tensor(Xf_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

    # 5. AIVC_Combo_Epistasis_Ens5
    ens_preds = []
    for sd_val in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd_val)
        net = ComboEpistasisOmniPathNet(Xf_tr.shape[1], 32, U_prog, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=5e-3)
        xa_t, xd_t, xf_t, xx_t, y_t = (
            torch.tensor(Xadd_tr, device=DEVICE),
            torch.tensor(Xdom_tr, device=DEVICE),
            torch.tensor(Xf_tr, device=DEVICE),
            torch.tensor(Xxt_tr, device=DEVICE),
            torch.tensor(Y_tr, device=DEVICE),
        )
        net.train()
        for _ in range(90):
            opt.zero_grad()
            pred = net(xa_t, xd_t, xf_t, xx_t)
            loss = F.mse_loss(pred, y_t) + 0.03 * (1.0 - F.cosine_similarity(pred, y_t, dim=-1)).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pev = net(
                torch.tensor(Xadd_ev, device=DEVICE),
                torch.tensor(Xdom_ev, device=DEVICE),
                torch.tensor(Xf_ev, device=DEVICE),
                torch.tensor(Xxt_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        ens_preds.append(pev)
        if sd_val == 42:
            print(f"  Learned combo weights (seed=42): w_add={net.w_add.item():.3f}, w_dom={net.w_dom.item():.3f}, alpha_xtalk={net.alpha_xtalk.item():.3f}")

    preds_dict = {
        "Train_Mean": p_mean,
        "Linear_Ridge": p_ridge,
        "Static_MLP": p_mlp,
        "Additive (dA+dB)": p_add,
        "AIVC_Combo_Epistasis_Ens5": np.mean(ens_preds, axis=0).astype(np.float32),
    }

    rows = []
    for mname, pmat in preds_dict.items():
        df_m = evaluate_perturbation_predictions(d_ev, {ev_keys[i]: pmat[i] for i in range(len(ev_keys))}, model_name=mname, top_k_list=[20])
        epi_r, epi_mse = eval_epistasis(Y_ev, pmat, Xadd_ev)
        row = {
            "model": mname,
            "mean_pearson": float(df_m["pearson_delta_all"].mean()),
            "top20_pearson": float(df_m["pearson_delta_top20"].mean()),
            "dir_acc_top20": float(df_m["dir_acc_top20"].mean()),
            "mse": float(df_m["mse_all"].mean()),
            "epistasis_r": epi_r,
            "epistasis_mse": epi_mse,
        }
        rows.append(row)
        print(
            f"  [{mname:26s}] MeanR={row['mean_pearson']:.4f} | Top20R={row['top20_pearson']:.4f} | "
            f"DirAcc={row['dir_acc_top20']:.4f} | MSE={row['mse']:.6f} | EpistasisR={row['epistasis_r']:.4f}"
        )

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=180)
    x = np.arange(len(res_df))
    w = 0.26
    axes[0].bar(x - w, res_df["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    axes[0].bar(x, res_df["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + w, res_df["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(" ", "\n").replace("_", "\n") for m in res_df["model"]], fontsize=8)
    axes[0].set_title("Unseen Double-Gene Combo Accuracy (Norman2019)", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8)

    colors_epi = ["#e74c3c" if "AIVC" in m else "#7f8c8d" for m in res_df["model"]]
    axes[1].bar(x, res_df["epistasis_r"], color=colors_epi, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(" ", "\n").replace("_", "\n") for m in res_df["model"]], fontsize=8)
    axes[1].set_ylabel("Epistasis Residual Pearson r (y - (dA+dB))")
    axes[1].set_title("Non-Additive Synergy / Epistasis Prediction", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp39 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
