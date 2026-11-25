# -*- coding: utf-8 -*-
"""
exp37_overexpression_joung_to_norman.py
Exp37: Pure Gene Overexpression (ORF + CRISPRa) Two-Stage Pretraining:
  Stage 1: Stream JoungZhang2023_atlas.h5ad (1.14M cells, ORF overexpression) in backed="r" mode,
           automatically infer each ORF's overexpressed TF target gene, and pretrain OverexpressionOmniPathNet.
  Stage 2: Fine-tune & blind-evaluate on Norman2019_hvg2000.h5ad Unseen-Gene CRISPRa Overexpression OOD split (80/20).
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
OUT_CSV = r"D:\data\AIVC\test_results\exp37_overexpression_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp37_overexpression_plot.png"


class OverexpressionOmniPathNet(nn.Module):
    """
    Gene Overexpression World Model:
      Uses ESM2 target gene embedding + OmniPath signed 1-2 hop activation causal footprint (+A_signed)
      + low-rank expression program basis U_prog.
    """

    def __init__(self, d_esm: int, k_prog: int, U_prog: np.ndarray, hid: int = 160):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.esm_enc = nn.Sequential(nn.Linear(d_esm, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.25))
        self.causal_enc = nn.Sequential(nn.Linear(k_prog, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25))
        self.head = nn.Sequential(
            nn.Linear(hid + 64, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, k_prog),
        )

    def forward(self, x_esm: torch.Tensor, x_causal: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.esm_enc(x_esm), self.causal_enc(x_causal)], dim=-1)
        return self.head(h) @ self.U_prog


def extract_joung_orf_deltas(target_genes: list, max_orfs: int = 450, max_cells_per_orf: int = 150):
    """Stream JoungZhang2023_atlas.h5ad in backed='r' mode to extract ORF overexpression deltas."""
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

    # '3368' is the largest control cluster (78,274 cells)
    ctrl_pos = np.where(obs_pert == "3368")[0]
    rng = np.random.RandomState(SEED)
    ctrl_sub = np.sort(rng.choice(ctrl_pos, size=min(4000, len(ctrl_pos)), replace=False))
    X_ctrl = a.X[ctrl_sub, :]
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = np.log1p(np.clip(np.asarray(X_ctrl, dtype=np.float32), 0.0, None))
    mu_ctrl_j = X_ctrl.mean(axis=0)
    mu_ctrl_shared = mu_ctrl_j[shared_j_idx]

    cand_orfs = [str(k) for k in vc.index if str(k) not in ("3368", "3367") and vc[k] >= 60][:max_orfs]
    orf_genes, orf_deltas = [], []
    for orf_id in cand_orfs:
        pos = np.where(obs_pert == orf_id)[0]
        if len(pos) > max_cells_per_orf:
            pos = np.sort(rng.choice(pos, size=max_cells_per_orf, replace=False))
        X_p = a.X[pos, :]
        if hasattr(X_p, "toarray"):
            X_p = X_p.toarray()
        X_p = np.log1p(np.clip(np.asarray(X_p, dtype=np.float32), 0.0, None))
        mu_p = X_p.mean(axis=0)
        # Identify overexpressed target gene across all Joung genes
        full_delta = mu_p - mu_ctrl_j
        top_j = int(np.argmax(full_delta))
        if full_delta[top_j] < 0.25:
            continue
        t_gene = j_genes[top_j].upper()
        dv_target = np.zeros(len(target_genes), dtype=np.float32)
        dv_target[shared_t_idx] = (mu_p[shared_j_idx] - mu_ctrl_shared).astype(np.float32)
        orf_genes.append(t_gene)
        orf_deltas.append(dv_target)
    a.file.close()
    mat = np.vstack(orf_deltas).astype(np.float32)
    print(f"  [JoungZhang2023 ORF] Extracted {len(mat)} verified TF ORF overexpression deltas (shared genes={len(shared_t_idx)}/{len(target_genes)})")
    return orf_genes, mat


def main():
    t0 = time.time()
    print(f"=== Exp37: Pure Gene Overexpression Pretraining (JoungZhang2023 -> Norman2019 CRISPRa OOD) ({DEVICE}) ===")

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
    print(f"  [Norman2019 CRISPRa] Single-gene overexpressions={len(single_perts)} -> Train={len(tr_perts)}, Unseen OOD Test={len(ev_perts)}")

    Y_tr = np.vstack([d_norm[p] for p in tr_perts]).astype(np.float32)
    Y_ev = np.vstack([d_norm[p] for p in ev_perts]).astype(np.float32)
    d_ev = {p: d_norm[p] for p in ev_perts}

    # Extract JoungZhang2023 ORF overexpression pretraining set (exclude any test gene to guarantee 0 leakage!)
    ev_set_up = {p.upper() for p in ev_perts}
    j_genes, Y_joung_raw = extract_joung_orf_deltas(target_genes, max_orfs=450, max_cells_per_orf=150)
    keep_j = [i for i, g in enumerate(j_genes) if g.upper() not in ev_set_up]
    j_genes = [j_genes[i] for i in keep_j]
    Y_joung = Y_joung_raw[keep_j]
    # Scale Joung log1p delta to match Norman2019 delta std
    Y_joung = Y_joung / (np.std(Y_joung) + 1e-6) * float(np.std(Y_tr))
    print(f"  [Zero-Leakage Filter] Joung ORFs after removing Norman test genes: {len(j_genes)}")

    esm2_dict = load_esm2_gene_embeddings(os.path.join(DATA_DIR, "target_esm2_150m.npz"))
    esm2_up = {str(k).upper(): v for k, v in esm2_dict.items()}
    d_esm = len(next(iter(esm2_up.values())))

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

    svd = TruncatedSVD(n_components=48, random_state=SEED).fit(np.vstack([Y_tr, Y_joung]))
    U_prog = svd.components_.astype(np.float32)
    # Overexpression causal footprint (+A_signed along OmniPath activation/repression)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

    def build_feats(plist):
        xes, xca = [], []
        for p in plist:
            pu = str(p).upper()
            xes.append(esm2_up.get(pu, np.zeros(d_esm, dtype=np.float32)))
            gi = gene_to_idx.get(pu, -1)
            xca.append(causal_gene_mat[gi] if gi >= 0 else np.zeros(48, dtype=np.float32))
        return np.asarray(xes, dtype=np.float32), np.asarray(xca, dtype=np.float32)

    X_esm_tr, X_cau_tr = build_feats(tr_perts)
    X_esm_ev, X_cau_ev = build_feats(ev_perts)
    X_esm_j, X_cau_j = build_feats(j_genes)

    rows = []
    # 1. Train-Mean
    p_mean = np.repeat(Y_tr.mean(axis=0, keepdims=True), len(ev_perts), axis=0).astype(np.float32)
    # 2. Ridge-ESM2
    ridge = Ridge(alpha=50.0).fit(X_esm_tr, Y_tr)
    p_ridge = ridge.predict(X_esm_ev).astype(np.float32)
    # 3. Static-MLP
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

    preds_dict = {"Train_Mean": p_mean, "Linear_Ridge_ESM2": p_ridge, "Static_MLP_ESM2": p_mlp}

    # 4 & 5: OverexpressionOmniPathNet (NoPretrain vs Joung2023 ORF Pretrained)
    for use_pre in [False, True]:
        tag = "AIVC_Overexpr_JoungPretrain" if use_pre else "AIVC_Overexpr_NoPretrain"
        torch.manual_seed(SEED)
        net = OverexpressionOmniPathNet(d_esm, 48, U_prog, hid=160).to(DEVICE)
        if use_pre:
            opt_p = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-3)
            dl_p = DataLoader(TensorDataset(torch.tensor(X_esm_j), torch.tensor(X_cau_j), torch.tensor(Y_joung)), batch_size=64, shuffle=True)
            net.train()
            for _ in range(40):
                for bx, bc, by in dl_p:
                    bx, bc, by = bx.to(DEVICE), bc.to(DEVICE), by.to(DEVICE)
                    opt_p.zero_grad()
                    F.mse_loss(net(bx, bc), by).backward()
                    opt_p.step()
        opt_f = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-3)
        net.train()
        xe_t, xc_t, y_t = torch.tensor(X_esm_tr, device=DEVICE), torch.tensor(X_cau_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
        for _ in range(80):
            opt_f.zero_grad()
            F.mse_loss(net(xe_t, xc_t), y_t).backward()
            opt_f.step()
        net.eval()
        with torch.no_grad():
            preds_dict[tag] = net(torch.tensor(X_esm_ev, device=DEVICE), torch.tensor(X_cau_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

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
        print(f"  [{mname:28s}] MeanR={row['mean_pearson']:.4f} | Top20R={row['top20_pearson']:.4f} | DirAcc={row['dir_acc_top20']:.4f} | MSE={row['mse']:.6f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(res_df))
    w = 0.26
    ax.bar(x - w, res_df["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, res_df["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, res_df["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in res_df["model"]], fontsize=8.5)
    ax.set_title("Exp37: Gene Overexpression Pretraining (JoungZhang2023 ORF -> Norman2019 Unseen CRISPRa)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp37 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
