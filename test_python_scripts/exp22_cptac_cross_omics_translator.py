import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
CANCERS = ["brca", "ccrcc", "coad", "gbm", "hnscc", "lscc", "luad", "ov", "pdac", "ucec"]
TRAIN_CANCERS = ["brca", "ccrcc", "coad", "gbm", "hnscc", "lscc", "pdac", "ucec"]
TEST_CANCERS = ["luad", "ov"]

OUT_CSV = r"D:\data\AIVC\test_results\exp22_cptac_cross_omics_results.csv"
OUT_TARGET_CSV = r"D:\data\AIVC\test_results\exp22_cptac_druggable_targets.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp22_cptac_cross_omics_plot.png"

KEY_DRUG_TARGETS = [
    "ERBB2", "EGFR", "CDK1", "CDK2", "CDK4", "AKT1", "MAPK1", "MTOR",
    "TOP2A", "PARP1", "AURKA", "BCL2", "EZH2", "HDAC1", "HDAC2", "MET", "PIK3CA", "TP53"
]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_cptac_cancer(cancer: str):
    """Load paired RNA, Proteome, and Gene-level Phosphoproteome for a CPTAC cancer type."""
    r_p = os.path.join(CPTAC_DIR, f"{cancer}_transcriptomics_bcm_processed_data.csv")
    p_p = os.path.join(CPTAC_DIR, f"{cancer}_proteomics_bcm_processed_data.csv")
    ph_p = os.path.join(CPTAC_DIR, f"{cancer}_phosphoproteomics_bcm_processed_data.csv")
    si_p = os.path.join(CPTAC_DIR, f"{cancer}_proteomics_bcm_sample_info.csv")

    rna = pd.read_csv(r_p, index_col=0)
    prot = pd.read_csv(p_p, index_col=0)
    phos = pd.read_csv(ph_p, index_col=0)
    si = pd.read_csv(si_p)

    # Collapse phosphosites GENE:SITE:ENSG -> GENE mean
    phos.index = [str(i).split(":")[0].strip().upper() for i in phos.index]
    phos = phos.groupby(phos.index, sort=False).mean()
    rna.index = [str(i).strip().upper() for i in rna.index]
    rna = rna.groupby(rna.index, sort=False).mean()
    prot.index = [str(i).strip().upper() for i in prot.index]
    prot = prot.groupby(prot.index, sort=False).mean()

    common_sids = sorted(set(rna.columns) & set(prot.columns) & set(phos.columns))
    si_map = dict(zip(si["sample.id"].astype(str), si["tissue.type"].astype(str)))
    labels = [si_map.get(s, "Unknown") for s in common_sids]

    return rna[common_sids], prot[common_sids], phos[common_sids], common_sids, labels


class CrossOmicsTranslator(nn.Module):
    """Cis-Residual + Trans-Context Translator: Proteome_g = w_cis * RNA_g + trans_context(RNA)."""
    def __init__(self, n_rna: int, n_prot: int, n_phos: int, prot_cis_idx: list, phos_cis_idx: list, hid: int = 256):
        super().__init__()
        self.register_buffer("prot_cis_idx", torch.tensor(prot_cis_idx, dtype=torch.long))
        self.register_buffer("phos_cis_idx", torch.tensor(phos_cis_idx, dtype=torch.long))
        self.prot_cis_w = nn.Parameter(torch.ones(len(prot_cis_idx)) * 0.5)
        self.phos_cis_w = nn.Parameter(torch.ones(len(phos_cis_idx)) * 0.25)

        self.rna_enc = nn.Sequential(
            nn.Linear(n_rna, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hid, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.prot_trans = nn.Linear(128, n_prot)
        self.phos_trans = nn.Linear(128, n_phos)
        nn.init.zeros_(self.prot_trans.weight)
        nn.init.zeros_(self.phos_trans.weight)

    def forward(self, x_rna):
        h = self.rna_enc(x_rna)
        prot_cis = torch.zeros((x_rna.shape[0], self.prot_trans.out_features), device=x_rna.device, dtype=x_rna.dtype)
        prot_cis = x_rna[:, self.prot_cis_idx] * self.prot_cis_w
        phos_cis = x_rna[:, self.phos_cis_idx] * self.phos_cis_w
        return prot_cis + 0.3 * self.prot_trans(h), phos_cis + 0.3 * self.phos_trans(h)


def main():
    print(f"=== Exp22: CPTAC 10-Cancer Cross-Omics (RNA -> Proteome & Phosphoproteome) ({DEVICE}) ===")
    loaded = {}
    shared_rna, shared_prot, shared_phos = None, None, None
    for c in CANCERS:
        rna, prot, phos, sids, labels = load_cptac_cancer(c)
        loaded[c] = (rna, prot, phos, sids, labels)
        shared_rna = set(rna.index) if shared_rna is None else (shared_rna & set(rna.index))
        shared_prot = set(prot.index) if shared_prot is None else (shared_prot & set(prot.index))
        shared_phos = set(phos.index) if shared_phos is None else (shared_phos & set(phos.index))
        print(f"  Loaded {c:6s}: PairedSamples={len(sids):3d} | RNA={len(rna)}, Prot={len(prot)}, PhosGenes={len(phos)}")

    # Select top 2,000 variable RNA genes, top 1,500 variable Proteins, top 1,000 variable Phospho-genes (preserving KEY_DRUG_TARGETS)
    brca_r, brca_p, brca_ph, _, _ = loaded["brca"]
    rna_cand = sorted(shared_rna)
    prot_cand = sorted(shared_prot)
    phos_cand = sorted(shared_phos)

    top_rna = list(brca_r.loc[rna_cand].var(axis=1).sort_values(ascending=False).index[:2000])
    top_prot = list(brca_p.loc[prot_cand].var(axis=1).sort_values(ascending=False).index[:1500])
    top_phos = list(brca_ph.loc[phos_cand].var(axis=1).sort_values(ascending=False).index[:1000])

    for g in KEY_DRUG_TARGETS:
        if g in shared_rna and g not in top_rna:
            top_rna.append(g)
        if g in shared_prot and g not in top_prot:
            top_prot.append(g)
        if g in shared_phos and g not in top_phos:
            top_phos.append(g)

    # Ensure every target protein / phosphoprotein has its cis-mRNA included if available in shared_rna
    for g in top_prot + top_phos:
        if g in shared_rna and g not in top_rna:
            top_rna.append(g)
    top_rna = sorted(top_rna)
    top_prot = sorted(g for g in top_prot if g in shared_rna)
    top_phos = sorted(g for g in top_phos if g in shared_rna)
    rna_pos = {g: i for i, g in enumerate(top_rna)}
    prot_cis_idx = [rna_pos[g] for g in top_prot]
    phos_cis_idx = [rna_pos[g] for g in top_phos]
    print(f"Shared across ALL 10 cancers -> Input RNA={len(top_rna)} | Target Prot={len(top_prot)} | Target Phos={len(top_phos)}")

    def build_split(cancer_list):
        xr_l, xp_l, xph_l, c_names = [], [], [], []
        for c in cancer_list:
            r, p, ph, sids, _ = loaded[c]
            xr = row_zscore(r.loc[top_rna].T.fillna(0.0).values)
            xp = row_zscore(p.loc[top_prot].T.fillna(0.0).values)
            xph = row_zscore(ph.loc[top_phos].T.fillna(0.0).values)
            xr_l.append(xr)
            xp_l.append(xp)
            xph_l.append(xph)
            c_names.extend([c] * len(sids))
        return np.vstack(xr_l), np.vstack(xp_l), np.vstack(xph_l), np.array(c_names)

    X_tr_r, Y_tr_p, Y_tr_ph, tr_c = build_split(TRAIN_CANCERS)
    X_te_r, Y_te_p, Y_te_ph, te_c = build_split(TEST_CANCERS)
    print(f"Train samples (8 cancers): {len(X_tr_r)} | Unseen Blind Test samples (LUAD + OV): {len(X_te_r)}")

    # Train CrossOmicsTranslator
    net = CrossOmicsTranslator(len(top_rna), len(top_prot), len(top_phos), prot_cis_idx, phos_cis_idx).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_r), torch.tensor(Y_tr_p), torch.tensor(Y_tr_ph)),
        batch_size=128, shuffle=True
    )
    net.train()
    for ep in range(40):
        tot = 0.0
        for bx, bp, bph in loader:
            bx, bp, bph = bx.to(DEVICE), bp.to(DEVICE), bph.to(DEVICE)
            opt.zero_grad()
            pp, pph = net(bx)
            loss = loss_fn(pp, bp) + loss_fn(pph, bph)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(bx)
        if (ep + 1) in (1, 10, 20, 40):
            print(f"  Epoch {ep+1:2d}/40 | Combined MSE={tot / len(X_tr_r):.4f}")

    net.eval()
    with torch.no_grad():
        pred_p, pred_ph = net(torch.tensor(X_te_r, device=DEVICE))
        pred_p = pred_p.cpu().numpy()
        pred_ph = pred_ph.cpu().numpy()

    # Evaluate per-gene cross-patient Pearson r on unseen LUAD & OV vs Naive 1-to-1 RNA baseline
    rna_idx_map = {g: i for i, g in enumerate(top_rna)}
    summary_rows = []
    target_rows = []

    for c_test in TEST_CANCERS:
        mask = (te_c == c_test)
        for mod_name, target_genes, y_true, y_pred in [
            ("Proteome", top_prot, Y_te_p[mask], pred_p[mask]),
            ("Phosphoproteome", top_phos, Y_te_ph[mask], pred_ph[mask]),
        ]:
            model_rs, naive_rs = [], []
            for j, g in enumerate(target_genes):
                yt = y_true[:, j]
                yp = y_pred[:, j]
                if np.std(yt) < 1e-6 or np.std(yp) < 1e-6:
                    continue
                rm, _ = pearsonr(yp, yt)
                model_rs.append(rm)
                if g in rna_idx_map:
                    rn, _ = pearsonr(X_te_r[mask, rna_idx_map[g]], yt)
                    naive_rs.append(rn)
                    if g in KEY_DRUG_TARGETS:
                        target_rows.append({
                            "cancer": c_test.upper(),
                            "modality": mod_name,
                            "gene": g,
                            "naive_rna_r": rn,
                            "translator_r": rm,
                            "gain": rm - rn,
                        })
            med_m, mean_m = float(np.median(model_rs)), float(np.mean(model_rs))
            med_n, mean_n = float(np.median(naive_rs)), float(np.mean(naive_rs))
            frac_pos = float(np.mean(np.array(model_rs) > 0.3))
            summary_rows.append({
                "unseen_cancer": c_test.upper(),
                "modality": mod_name,
                "n_patients": int(mask.sum()),
                "n_features": len(model_rs),
                "naive_1to1_median_r": med_n,
                "translator_median_r": med_m,
                "naive_1to1_mean_r": mean_n,
                "translator_mean_r": mean_m,
                "pct_genes_r_gt_0.3": frac_pos * 100,
            })
            print(f"  [BLIND {c_test.upper()} | {mod_name:15s}] Naive 1:1 Med r={med_n:.4f} | AIVC Translator Med r={med_m:.4f} (Mean={mean_m:.4f}, >0.3: {frac_pos*100:.1f}%)")

    sum_df = pd.DataFrame(summary_rows)
    targ_df = pd.DataFrame(target_rows).sort_values(["cancer", "modality", "translator_r"], ascending=[True, True, False])
    sum_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    targ_df.to_csv(OUT_TARGET_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Top Druggable Targets Predicted from RNA Alone (Unseen LUAD & OV) ===")
    print(targ_df[targ_df["modality"] == "Proteome"].head(15).to_string(index=False))

    # Plot 2-panel comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    labels_x = [f"{r['unseen_cancer']}\n{r['modality']}" for _, r in sum_df.iterrows()]
    xp = np.arange(len(sum_df))
    axes[0].bar(xp - 0.18, sum_df["naive_1to1_median_r"], width=0.35, label="Naive 1-to-1 mRNA Baseline", color="#95a5a6", edgecolor="black", linewidth=0.5)
    axes[0].bar(xp + 0.18, sum_df["translator_median_r"], width=0.35, label="AIVC Cross-Omics Translator", color="#2980b9", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(labels_x, fontsize=9)
    axes[0].set_ylabel("Median Cross-Patient Pearson (r)")
    axes[0].set_title("Blind Unseen Cancer Types (LUAD & OV)\nRNA -> Virtual Proteome & Phosphoproteome", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8.5)

    luad_t = targ_df[(targ_df["cancer"] == "LUAD") & (targ_df["modality"] == "Proteome")].head(10)
    yp = np.arange(len(luad_t))
    axes[1].barh(yp - 0.18, luad_t["naive_rna_r"], height=0.35, label="Naive Same-Gene mRNA", color="#bdc3c7", edgecolor="black", linewidth=0.5)
    axes[1].barh(yp + 0.18, luad_t["translator_r"], height=0.35, label="AIVC Virtual Proteome", color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(yp)
    axes[1].set_yticklabels(luad_t["gene"], fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cross-Patient Pearson (r) in Unseen LUAD")
    axes[1].set_title("Key Druggable Targets Protein Prediction\nfrom Transcriptome Alone (Unseen LUAD)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp22 figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
