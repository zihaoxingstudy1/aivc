import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from bfc import build_gene_coexp_prior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_CSV = r"D:\data\AIVC\test_results\exp29_cross_lineage_rpe1_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp29_cross_lineage_rpe1_plot.png"


class LowRankBaselineGatedNet(nn.Module):
    """
    Exp29 Cross-Lineage Perturbation Model:
      - Explicit Target-Gene Baseline Expression Gate: modulates amplitude by how strongly
        the perturbed gene g is actually expressed in the target cell line C!
      - Gene-Program Low-Rank Bottleneck (K=64): predicts residual adjustments in shared
        biological program space rather than 2,000 noisy independent genes.
      - Learnable per-gene scaling on the Linear-Ridge cross-lineage prior.
    """
    def __init__(self, n_genes: int, n_teacher: int, U_prog: np.ndarray, hid: int = 128):
        super().__init__()
        # U_prog: (K, n_genes) orthonormal gene-program basis
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        k_prog = U_prog.shape[0]

        self.ridge_scale = nn.Parameter(torch.ones(1) * 1.0)
        # Explicit target-gene baseline expression gate: [x_ctrl[g], z_ctrl[g], ridge_norm] -> scalar/program gate
        self.target_gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.prog_head = nn.Sequential(
            nn.Linear(64 + hid + 3, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.prog_head[-1].weight)
        nn.init.zeros_(self.prog_head[-1].bias)

    def forward(self, x_ctrl, x_teach, x_ridge, x_tstat):
        # x_tstat: (B, 3) = [raw_target_expr, z_target_expr, log_ridge_norm]
        gate = 1.0 + 0.5 * torch.tanh(self.target_gate(x_tstat))  # (B, 1) around 1.0
        h = torch.cat([self.ctrl_enc(x_ctrl), self.teach_enc(x_teach), x_tstat], dim=-1)
        prog_coeff = self.prog_head(h)  # (B, K)
        delta_prog = prog_coeff @ self.U_prog  # (B, n_genes)
        return gate * (self.ridge_scale * x_ridge) + 0.5 * delta_prog


def main():
    print(f"=== Exp29: Conquering Cross-Lineage (7-Lineage -> Unseen RPE1) ({DEVICE}) ===")
    ref_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    ref = sc.read_h5ad(ref_h5ad)
    target_genes = list(ref.var_names)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}

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
    for lname, mc, dd in lineage_specs:
        for p, dv in dd.items():
            if "_" in p or "+" in p:
                continue
            tr_samples.append((lname, p, mc, dv))

    ev_keys = sorted(d_rpe1.keys())
    print(f"\nTrain samples: {len(tr_samples)} across {len(lineage_specs)} lineages -> Blind RPE1: {len(ev_keys)} perts")

    # 1. Baseline Linear-Ridge(7-Lineage)
    tr_vocab = sorted({s[1] for s in tr_samples})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    X_tr_lin = enc_lin([s[1] for s in tr_samples])
    Y_tr_lin = np.vstack([s[3] for s in tr_samples])

    # Fit Ridge (with out-of-fold or regularized prediction for training residual)
    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_lin)
    pred_ridge_ev = ridge.predict(enc_lin(ev_keys))
    df_ridge = evaluate_perturbation_predictions(d_rpe1, {k: pred_ridge_ev[i] for i, k in enumerate(ev_keys)}, "Linear-Ridge(7-Lineage)")

    # 2. Extract Gene-Program Low-Rank Basis (K=64) from Training Deltas
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    svd.fit(Y_tr_lin)
    U_prog = svd.components_.astype(np.float32)  # (64, 2000)

    # 3. Build Teacher Feature Matrix (Coexp 64 + ESM2) + Target Baseline Expression Stats
    esm_dim = len(next(iter(esm2_dict.values())))
    co_dim = len(next(iter(coexp_dict.values())))

    def get_teach_vec(p: str) -> np.ndarray:
        c = coexp_dict.get(p, np.zeros(co_dim, dtype=np.float32))
        e = esm2_dict.get(p, np.zeros(esm_dim, dtype=np.float32))
        return np.concatenate([c, e]).astype(np.float32)

    def get_target_stats(mc: np.ndarray, p: str, r_vec: np.ndarray) -> np.ndarray:
        mu_m = float(mc.mean())
        sd_m = max(float(mc.std()), 1e-6)
        if p in gene_to_idx:
            raw_e = float(mc[gene_to_idx[p]])
            z_e = (raw_e - mu_m) / sd_m
        else:
            raw_e, z_e = mu_m, 0.0
        r_norm = float(np.log1p(np.linalg.norm(r_vec)))
        return np.array([raw_e, z_e, r_norm], dtype=np.float32)

    # Also compute cross-lineage Leave-One-Lineage-Out Ridge for training so net learns true transfer residual!
    pred_ridge_tr = np.zeros_like(Y_tr_lin)
    lineages_tr = np.array([s[0] for s in tr_samples])
    for ln in np.unique(lineages_tr):
        m_out = (lineages_tr == ln)
        m_in = ~m_out
        if m_in.sum() > 0:
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[m_in], Y_tr_lin[m_in])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out])

    X_tr_ctrl = np.vstack([s[2] for s in tr_samples]).astype(np.float32)
    X_tr_teach = np.vstack([get_teach_vec(s[1]) for s in tr_samples])
    X_tr_tstat = np.vstack([get_target_stats(s[2], s[1], pred_ridge_tr[i]) for i, s in enumerate(tr_samples)])

    X_ev_ctrl = np.tile(mu_rpe1, (len(ev_keys), 1)).astype(np.float32)
    X_ev_teach = np.vstack([get_teach_vec(k) for k in ev_keys])
    X_ev_tstat = np.vstack([get_target_stats(mu_rpe1, k, pred_ridge_ev[i]) for i, k in enumerate(ev_keys)])

    # Train LowRankBaselineGatedNet
    net = LowRankBaselineGatedNet(len(target_genes), X_tr_teach.shape[1], U_prog).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr_ctrl), torch.tensor(X_tr_teach),
            torch.tensor(pred_ridge_tr), torch.tensor(X_tr_tstat), torch.tensor(Y_tr_lin)
        ),
        batch_size=256, shuffle=True
    )
    net.train()
    for ep in range(45):
        for bc, bt, br, bs, by in loader:
            bc, bt, br, bs, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bs.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            pred = net(bc, bt, br, bs)
            loss = F.mse_loss(pred, by) + 0.05 * (1.0 - F.cosine_similarity(pred, by, dim=-1)).mean()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred_exp29 = net(
            torch.tensor(X_ev_ctrl, device=DEVICE),
            torch.tensor(X_ev_teach, device=DEVICE),
            torch.tensor(pred_ridge_ev, device=DEVICE),
            torch.tensor(X_ev_tstat, device=DEVICE),
        ).cpu().numpy()

    df_exp29 = evaluate_perturbation_predictions(d_rpe1, {k: pred_exp29[i] for i, k in enumerate(ev_keys)}, "Exp29_LowRank_TargetGated")

    # Compare summary
    comp = pd.DataFrame([
        {
            "model": "Linear-Ridge(7-Lineage)",
            "mean_pearson": df_ridge["pearson_delta_all"].mean(),
            "top20_pearson": df_ridge["pearson_delta_top20"].mean(),
            "dir_acc_top20": df_ridge["dir_acc_top20"].mean(),
            "mse": df_ridge["mse_all"].mean(),
        },
        {
            "model": "Exp29_LowRank_TargetGated",
            "mean_pearson": df_exp29["pearson_delta_all"].mean(),
            "top20_pearson": df_exp29["pearson_delta_top20"].mean(),
            "dir_acc_top20": df_exp29["dir_acc_top20"].mean(),
            "mse": df_exp29["mse_all"].mean(),
        },
    ])
    comp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n=== Exp29 vs Linear-Ridge on Blind Unseen RPE1 (2,016 perturbations) ===")
    print(comp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    x = np.arange(2)
    ax.bar(x - 0.16, comp["mean_pearson"], width=0.32, label="All-Gene Mean Pearson", color="#2980b9", edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.16, comp["top20_pearson"], width=0.32, label="Top-20 DEG Pearson", color="#e67e22", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Pearson Correlation on Unseen RPE1")
    ax.set_title("Exp29: Conquering Cross-Lineage Zero-Shot Transfer\n(7-Lineage -> Unseen RPE1, 2,016 Perturbations)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp29 results to {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
