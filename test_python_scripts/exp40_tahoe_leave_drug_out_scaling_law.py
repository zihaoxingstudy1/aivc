# -*- coding: utf-8 -*-
"""
exp40_tahoe_leave_drug_out_scaling_law.py
Exp40: Tahoe-100M Strict Leave-Drug-Out (Unseen New Molecule OOD) & Data Scaling Law Benchmark
  - Streams Tahoe-100M pseudobulk differential expression across ~360 drugs (aligned to 2,000 HVGs).
  - Holds out 20% completely UNSEEN DRUGS (never seen during training) as strict Leave-Drug-Out OOD test set.
  - Compares Train-Mean, Linear-Ridge (ChemBERTa+Morgan), Static-MLP, and
    AIVC_Drug_OmniPath (ChemBERTa+Morgan + Known-Target OmniPath 1-2 Hop Causal Footprint + U_prog).
  - Measures Data Scaling Law across 10%, 25%, 50%, 100% of training drugs.
"""

import os
import sys
import glob
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
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from pgnn import build_gene_pathway_bipartite_tensors
from chem_interface import encode_smiles_multimodal

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
TAHOE_META_DIR = r"D:\data\AIVC\data\tahoe_100m\metadata"
TAHOE_PB_DIR = os.path.join(TAHOE_META_DIR, "pseudobulk_differential_expression")
CHEMBERTA_DIR = r"D:\data\AIVC\pretrained\chemberta"

OUT_CSV = r"D:\data\AIVC\test_results\exp40_leave_drug_out_scaling_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp40_leave_drug_out_scaling_plot.png"


class DrugOmniPathCausalNet(nn.Module):
    """
    Zero-Shot Unseen-Drug World Model:
      Combines ChemBERTa+Morgan continuous structure embedding + Target-to-OmniPath 1-2 hop causal footprint
      + Ridge residual anchor on low-rank drug-response program basis U_prog.
    """

    def __init__(self, d_mol: int, k_prog: int, U_prog: np.ndarray, hid: int = 192):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.mol_enc = nn.Sequential(nn.Linear(d_mol, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.3))
        self.cau_enc = nn.Sequential(nn.Linear(k_prog, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25))
        self.head = nn.Sequential(
            nn.Linear(hid + 64, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_mol, x_cau, x_ridge):
        h = torch.cat([self.mol_enc(x_mol), self.cau_enc(x_cau)], dim=-1)
        return self.ridge_scale * x_ridge + 0.5 * (self.head(h) @ self.U_prog)


def eval_drug_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p_all, p_top20, d_top20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p_all.append(float(pearsonr(yt, yp)[0]))
        top_idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[top_idx]) > 1e-6 and np.std(yp[top_idx]) > 1e-6:
            p_top20.append(float(pearsonr(yt[top_idx], yp[top_idx])[0]))
        d_top20.append(float((np.sign(yt[top_idx]) == np.sign(yp[top_idx])).mean()))
    return {
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def main():
    t0 = time.time()
    print(f"=== Exp40: Tahoe-100M Strict Leave-Drug-Out OOD & Data Scaling Law ({DEVICE}) ===")

    ad_mix = sc.read_h5ad(os.path.join(DATA_DIR, "MIXseq_hvg2000.h5ad"), backed="r")
    genes = [str(g) for g in ad_mix.var_names]
    ad_mix.file.close()
    n_genes = len(genes)
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    gset = set(g2i.keys())

    drug_meta = pd.read_parquet(os.path.join(TAHOE_META_DIR, "drug_metadata.parquet"))
    smi_map, tgt_map = {}, {}
    for d, smi, tgts in zip(drug_meta["drug"], drug_meta["canonical_smiles"], drug_meta["targets"]):
        if pd.notna(d) and pd.notna(smi) and str(smi).strip():
            ds = str(d).strip()
            smi_map[ds] = str(smi).strip()
            if pd.notna(tgts) and str(tgts).strip():
                tgt_map[ds] = [t.strip().upper() for t in str(tgts).replace(";", ",").split(",") if t.strip()]
            else:
                tgt_map[ds] = []

    # Stream 35 shards across Tahoe-100M and aggregate mean drug-level signature across cell lines (consensus drug response)
    shards = sorted(glob.glob(os.path.join(TAHOE_PB_DIR, "train-*.parquet")))[:35]
    drug_sums = {}
    drug_cnts = {}
    for sp in shards:
        df = pd.read_parquet(sp, columns=["gene_name", "log2FoldChange", "Cell_Name_Vevo", "drug", "concentration"])
        df["g_up"] = df["gene_name"].astype(str).str.upper()
        df = df[df["g_up"].isin(gset)].dropna(subset=["log2FoldChange"])
        for (cl, drg, conc), grp in df.groupby(["Cell_Name_Vevo", "drug", "concentration"], sort=False):
            ds = str(drg).strip()
            if ds not in smi_map:
                continue
            v = np.zeros(n_genes, dtype=np.float32)
            idxs = [g2i[g] for g in grp["g_up"].values]
            v[idxs] = np.clip(grp["log2FoldChange"].values.astype(np.float32), -4.0, 4.0)
            if ds not in drug_sums:
                drug_sums[ds] = v.astype(np.float64)
                drug_cnts[ds] = 1
            else:
                drug_sums[ds] += v
                drug_cnts[ds] += 1

    all_drugs = sorted([d for d, c in drug_cnts.items() if c >= 1])
    Y_all = np.vstack([(drug_sums[d] / drug_cnts[d]).astype(np.float32) for d in all_drugs])
    print(f"  [Tahoe-100M Consensus] Extracted {len(all_drugs)} unique drugs from {len(shards)} shards (total cell-drug profiles={sum(drug_cnts.values())})")

    morg, cb = encode_smiles_multimodal([smi_map[d] for d in all_drugs], chemberta_dir=CHEMBERTA_DIR, device_str="cpu")
    X_mol_all = np.hstack([cb, morg]).astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A2 = A_signed @ A_signed

    # Strict Leave-Drug-Out split: 80% Train Drugs -> 20% Completely Unseen Test Drugs
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(all_drugs))
    n_tr_total = int(0.8 * len(all_drugs))
    tr_idx_full = perm[:n_tr_total]
    ev_idx = perm[n_tr_total:]
    print(f"  [Leave-Drug-Out Split] Train Drugs={len(tr_idx_full)}, Unseen Test Drugs={len(ev_idx)}")

    Y_ev = Y_all[ev_idx]
    X_mol_ev = X_mol_all[ev_idx]

    svd = TruncatedSVD(n_components=48, random_state=SEED).fit(Y_all[tr_idx_full])
    U_prog = svd.components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * A2) @ U_prog.T).astype(np.float32)

    def build_drug_causal(indices):
        out = []
        for idx in indices:
            d = all_drugs[idx]
            tgts = [g2i[t] for t in tgt_map.get(d, []) if t in g2i]
            if tgts:
                out.append(np.mean([causal_gene_mat[gi] for gi in tgts], axis=0))
            else:
                out.append(np.zeros(48, dtype=np.float32))
        return np.asarray(out, dtype=np.float32)

    X_cau_ev = build_drug_causal(ev_idx)

    rows = []
    # Baselines at 100% training drugs
    X_tr_100, Y_tr_100 = X_mol_all[tr_idx_full], Y_all[tr_idx_full]
    p_mean = np.repeat(Y_tr_100.mean(axis=0, keepdims=True), len(ev_idx), axis=0).astype(np.float32)
    ridge_100 = Ridge(alpha=200.0).fit(X_tr_100, Y_tr_100)
    p_ridge_100 = ridge_100.predict(X_mol_ev).astype(np.float32)

    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(X_tr_100.shape[1], 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.3), nn.Linear(192, n_genes)).to(DEVICE)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=3e-3)
    for _ in range(60):
        opt_m.zero_grad()
        F.mse_loss(mlp(torch.tensor(X_tr_100, device=DEVICE)), torch.tensor(Y_tr_100, device=DEVICE)).backward()
        opt_m.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp_100 = mlp(torch.tensor(X_mol_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

    for mname, pmat in [("Train_Mean", p_mean), ("Linear_Ridge_SMILES", p_ridge_100), ("Static_MLP_SMILES", p_mlp_100)]:
        m = eval_drug_metrics(Y_ev, pmat)
        print(f"  [{mname:28s} | frac=100%] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
        rows.append({"model": mname, "train_frac": 1.0, "n_train_drugs": len(tr_idx_full), **m})

    # Scaling Law for AIVC_Drug_OmniPath_Ens5 across 10%, 25%, 50%, 100% of training drugs
    for frac in [0.10, 0.25, 0.50, 1.00]:
        n_sub = max(20, int(round(len(tr_idx_full) * frac)))
        sub_idx = tr_idx_full[:n_sub]
        Xm_tr, Ym_tr = X_mol_all[sub_idx], Y_all[sub_idx]
        Xc_tr = build_drug_causal(sub_idx)

        # Out-of-fold Ridge on sub_idx
        r_oof = np.zeros_like(Ym_tr)
        n_k = min(5, max(2, n_sub // 10))
        fold_size = n_sub // n_k
        for fi in range(n_k):
            va = np.arange(fi * fold_size, n_sub if fi == n_k - 1 else (fi + 1) * fold_size)
            tr = np.setdiff1d(np.arange(n_sub), va)
            r_cv = Ridge(alpha=200.0).fit(Xm_tr[tr], Ym_tr[tr])
            r_oof[va] = r_cv.predict(Xm_tr[va]).astype(np.float32)
        r_full = Ridge(alpha=200.0).fit(Xm_tr, Ym_tr)
        r_ev_pred = r_full.predict(X_mol_ev).astype(np.float32)

        ens_p = []
        for sd_val in [42, 101, 202, 303, 404]:
            torch.manual_seed(sd_val)
            net = DrugOmniPathCausalNet(Xm_tr.shape[1], 48, U_prog, hid=192).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-3)
            xm_t = torch.tensor(Xm_tr, device=DEVICE)
            xc_t = torch.tensor(Xc_tr, device=DEVICE)
            xr_t = torch.tensor(r_oof, device=DEVICE)
            y_t = torch.tensor(Ym_tr, device=DEVICE)
            net.train()
            for _ in range(55):
                opt.zero_grad()
                pred = net(xm_t, xc_t, xr_t)
                loss = F.mse_loss(pred, y_t) + 0.04 * (1.0 - F.cosine_similarity(pred, y_t, dim=-1)).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pev = net(
                    torch.tensor(X_mol_ev, device=DEVICE),
                    torch.tensor(X_cau_ev, device=DEVICE),
                    torch.tensor(r_ev_pred, device=DEVICE),
                ).cpu().numpy().astype(np.float32)
            ens_p.append(pev)
        p_ens = np.mean(ens_p, axis=0).astype(np.float32)
        m = eval_drug_metrics(Y_ev, p_ens)
        mname = "AIVC_Drug_OmniPath_Ens5"
        print(f"  [{mname:28s} | frac={int(frac*100):3d}%] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
        rows.append({"model": mname, "train_frac": frac, "n_train_drugs": n_sub, **m})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)
    # Panel 1: Leave-Drug-Out Comparison at 100%
    sub100 = res_df[res_df["train_frac"] == 1.0]
    x = np.arange(len(sub100))
    w = 0.26
    axes[0].bar(x - w, sub100["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    axes[0].bar(x, sub100["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + w, sub100["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in sub100["model"]], fontsize=8)
    axes[0].set_title(f"Leave-Drug-Out Zero-Shot Benchmark\n({len(ev_idx)} Completely Unseen Drugs)", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8)

    # Panel 2: Data Scaling Law Curve
    sc_df = res_df[res_df["model"] == "AIVC_Drug_OmniPath_Ens5"].sort_values("n_train_drugs")
    axes[1].plot(sc_df["n_train_drugs"], sc_df["mean_pearson"], marker="o", linewidth=2.0, label="Mean Pearson r", color="#3498db")
    axes[1].plot(sc_df["n_train_drugs"], sc_df["top20_pearson"], marker="s", linewidth=2.0, label="Top20 DEG Pearson r", color="#2ecc71")
    axes[1].set_xlabel("Number of Training Drugs (Tahoe-100M)", fontsize=9.5)
    axes[1].set_ylabel("Unseen-Drug Pearson r", fontsize=9.5)
    axes[1].set_title("Data Scaling Law on Unseen Small-Molecule Drugs", fontsize=10.5, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp40 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
