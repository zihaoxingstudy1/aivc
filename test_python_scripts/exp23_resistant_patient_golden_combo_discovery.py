import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_CSV = r"D:\data\AIVC\test_results\exp23_resistant_patient_golden_combos.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp23_golden_combo_pareto.png"

VITAL_ORGANS = [
    "Whole Blood",
    "Liver",
    "Heart - Left Ventricle",
    "Kidney - Cortex",
    "Lung",
    "Brain - Cortex",
    "Nerve - Tibial",
    "Colon - Transverse",
]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class BioMolKGSynergyNet(nn.Module):
    def __init__(self, n_genes: int, n_drug: int = 2048, hid: int = 256):
        super().__init__()
        self.c_enc = nn.Sequential(nn.Linear(n_genes, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.d_enc = nn.Sequential(nn.Linear(n_drug, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128 * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, xc, xd1, xd2):
        hc = self.c_enc(xc)
        hd1 = self.d_enc(xd1)
        hd2 = self.d_enc(xd2)
        return self.head(torch.cat([hc, hd1 + hd2, hd1 * hd2], dim=-1)).squeeze(-1)


def main():
    print(f"=== Exp23: Golden Therapeutic-Window Combo Discovery for Chemoresistant Patients ({DEVICE}) ===")

    # 1. Load Resistant Patients (GSE25066 Microarray RD + GSE163882 RNA-seq RD) & GTEx Vital Organs
    c250 = pd.read_csv(r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE25066\GSE25066_clinical.csv")
    e250 = pd.read_parquet(r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE25066\GSE25066_expr_log2.parquet")
    rd250 = c250.loc[c250["response_binary"] == 0.0, "sample_id"].tolist()

    c163 = pd.read_csv(r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE163882\GSE163882_clinical.csv")
    e163 = pd.read_parquet(r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE163882\GSE163882_expr_log2.parquet")
    rd163 = c163.loc[c163["response_binary"] == 0.0, "sample_id"].tolist()

    gtex_df = pd.read_csv(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", sep="\t", skiprows=2)
    gtex_df["Description"] = gtex_df["Description"].astype(str).str.strip().str.upper()
    organ_cols = [c for c in gtex_df.columns if c not in ("Name", "Description")]
    gtex_gene = gtex_df.groupby("Description", sort=False)[organ_cols].mean()
    gtex_log2_df = pd.DataFrame(np.log2(np.clip(gtex_gene.values, 0.0, None) + 1.0).T, index=organ_cols, columns=gtex_gene.index)

    rna_raw = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", skiprows=[1, 2], low_memory=False)
    rna_raw = rna_raw.rename(columns={rna_raw.columns[0]: "gene_symbol"})
    rna_raw = rna_raw.drop(columns=[c for c in rna_raw.columns if str(c).startswith("Unnamed:")])
    rna_raw["gene_symbol"] = rna_raw["gene_symbol"].astype(str).str.strip().str.upper()

    shared_genes = sorted(set(e250.columns) & set(e163.columns) & set(gtex_log2_df.columns) & set(rna_raw["gene_symbol"]))
    rna_raw = rna_raw[rna_raw["gene_symbol"].isin(shared_genes)].drop_duplicates(subset=["gene_symbol"]).set_index("gene_symbol").apply(pd.to_numeric, errors="coerce")
    top_2000 = sorted(list(rna_raw.var(axis=1).sort_values(ascending=False).index[:2000]))
    print(f"Shared genes: {len(shared_genes)} | Model genes: {len(top_2000)} | Resistant RD patients: {len(rd250)} + {len(rd163)} = {len(rd250)+len(rd163)}")

    cell_expr_df = rna_raw.loc[top_2000].T.dropna(how="all")
    cell_z = row_zscore(cell_expr_df.values)
    cell_id_to_row = {sid: i for i, sid in enumerate(cell_expr_df.index)}

    # Representative resistant patient Z-score states (subsample 60 diverse RD patients across Microarray + RNA-seq for fast batch screening)
    z_rd_all = np.vstack([
        row_zscore(e250.loc[rd250[:40], top_2000].values),
        row_zscore(e163.loc[rd163[:40], top_2000].values),
    ])
    z_vital = row_zscore(gtex_log2_df.loc[VITAL_ORGANS, top_2000].values)

    # 2. Train BioMolKG-Synergy on DrugComb v1.5 (310k real wet-lab combos)
    c_npz = np.load(r"D:\data\AIVC\data\combo_drug_kg_features.npz", allow_pickle=True)
    combo_d2f = {str(d).upper(): c_npz["feats"][i].astype(np.float32) for i, d in enumerate(c_npz["drugs"])}
    combo_d2orig = {str(d).upper(): str(d) for d in c_npz["drugs"]}

    cmp = pd.read_csv(r"D:\data\AIVC\data\cmp_model_list.csv", usecols=["model_id", "model_name"]).dropna()
    cname_to_row = {str(mn).strip().upper(): cell_id_to_row[mid] for mid, mn in zip(cmp["model_id"], cmp["model_name"]) if mid in cell_id_to_row}

    dc = pd.read_csv(r"D:\data\drug\summary_v_1_5.csv", usecols=["drug_row", "drug_col", "cell_line_name", "synergy_bliss"], low_memory=False).dropna()
    dc["d1u"] = dc["drug_row"].astype(str).str.strip().str.upper()
    dc["d2u"] = dc["drug_col"].astype(str).str.strip().str.upper()
    dc["cu"] = dc["cell_line_name"].astype(str).str.strip().str.upper()
    dc = dc[dc["d1u"].isin(combo_d2f) & dc["d2u"].isin(combo_d2f) & dc["cu"].isin(cname_to_row)].copy()
    dc["synergy_bliss"] = np.clip(pd.to_numeric(dc["synergy_bliss"], errors="coerce").values, -50.0, 50.0).astype(np.float32)
    dc = dc.dropna(subset=["synergy_bliss"])

    # Pick Top 60 most frequently tested oncology/repurposing drugs in DrugComb for pairwise screening (60*59/2 = 1,770 combos)
    drug_counts = pd.concat([dc["d1u"], dc["d2u"]]).value_counts()
    top_drugs = [d for d in drug_counts.index[:60]]

    sc_idx = dc["cu"].map(cname_to_row).values
    all_u = sorted(set(dc["d1u"]) | set(dc["d2u"]))
    u2i = {u: i for i, u in enumerate(all_u)}
    sd1_idx = dc["d1u"].map(u2i).values
    sd2_idx = dc["d2u"].map(u2i).values
    sd_mat = np.stack([combo_d2f[u] for u in all_u], axis=0)
    y_bliss = dc["synergy_bliss"].values

    syn_net = BioMolKGSynergyNet(len(top_2000), 2048).to(DEVICE)
    opt = torch.optim.AdamW(syn_net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    cell_z_t = torch.tensor(cell_z, device=DEVICE)
    sd_mat_t = torch.tensor(sd_mat, device=DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(sc_idx), torch.tensor(sd1_idx), torch.tensor(sd2_idx), torch.tensor(y_bliss)), batch_size=2048, shuffle=True)
    print(f"Training BioMolKG-Synergy on {len(y_bliss)} DrugComb combos...")
    syn_net.train()
    for ep in range(6):
        tot = 0.0
        for bc, bd1, bd2, by in loader:
            opt.zero_grad()
            loss = loss_fn(syn_net(cell_z_t[bc], sd_mat_t[bd1], sd_mat_t[bd2]), by.to(DEVICE))
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(by)
        print(f"  Epoch {ep+1}/6 | MSE={tot / len(y_bliss):.4f}")

    # 3. Screen all 1,770 candidate drug pairs across 80 Chemoresistant RD Patients & 8 Vital GTEx Organs
    syn_net.eval()
    pairs = [(top_drugs[i], top_drugs[j]) for i in range(len(top_drugs)) for j in range(i + 1, len(top_drugs))]
    print(f"Screening {len(pairs)} candidate combinations across 80 Chemoresistant RD patients & 8 Vital Organs...")

    with torch.no_grad():
        hc_rd = syn_net.c_enc(torch.tensor(z_rd_all, device=DEVICE))       # (80, 128)
        hc_vit = syn_net.c_enc(torch.tensor(z_vital, device=DEVICE))       # (8, 128)
        top_feats = torch.tensor(np.stack([combo_d2f[d] for d in top_drugs]), device=DEVICE)
        hd_top = syn_net.d_enc(top_feats)                                  # (60, 128)

        # Evaluate in chunks of pairs
        pair_idx = [(i, j) for i in range(len(top_drugs)) for j in range(i + 1, len(top_drugs))]
        rows = []
        for idx_p, (i, j) in enumerate(pair_idx):
            d1_h, d2_h = hd_top[i], hd_top[j]
            s_sum = (d1_h + d2_h).unsqueeze(0)
            s_prod = (d1_h * d2_h).unsqueeze(0)

            rd_in = torch.cat([hc_rd, s_sum.expand(len(hc_rd), -1), s_prod.expand(len(hc_rd), -1)], dim=-1)
            vit_in = torch.cat([hc_vit, s_sum.expand(len(hc_vit), -1), s_prod.expand(len(hc_vit), -1)], dim=-1)

            rd_scores = syn_net.head(rd_in).squeeze(-1).cpu().numpy()
            vit_scores = syn_net.head(vit_in).squeeze(-1).cpu().numpy()

            rd_mean = float(np.mean(rd_scores))
            vit_max = float(np.max(vit_scores))
            vit_mean = float(np.mean(vit_scores))
            worst_org = VITAL_ORGANS[int(np.argmax(vit_scores))]
            rows.append({
                "drug_1": combo_d2orig[top_drugs[i]],
                "drug_2": combo_d2orig[top_drugs[j]],
                "combo_pair": f"{combo_d2orig[top_drugs[i]]} + {combo_d2orig[top_drugs[j]]}",
                "rd_patient_synergy_bliss": rd_mean,
                "vital_organ_mean_bliss": vit_mean,
                "vital_organ_max_bliss": vit_max,
                "worst_organ": worst_org,
                "therapeutic_window_score": rd_mean - max(0.0, vit_max),
            })

    res_df = pd.DataFrame(rows).sort_values("therapeutic_window_score", ascending=False).reset_index(drop=True)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n=== TOP 15 GOLDEN THERAPEUTIC-WINDOW RESCUE COMBOS FOR CHEMORESISTANT PATIENTS ===")
    print(res_df.head(15)[["combo_pair", "rd_patient_synergy_bliss", "vital_organ_max_bliss", "worst_organ", "therapeutic_window_score"]].to_string(index=False))

    # Plot Pareto Frontier: Resistant Patient Synergy vs Vital Organ Max Toxicity Synergy
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
    sc = ax.scatter(
        res_df["vital_organ_max_bliss"],
        res_df["rd_patient_synergy_bliss"],
        c=res_df["therapeutic_window_score"],
        cmap="RdYlGn",
        s=28,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.3,
    )
    plt.colorbar(sc, label="Golden Therapeutic Window Score")
    # Annotate Top 8 Golden Combos
    for idx_t in range(8):
        r = res_df.iloc[idx_t]
        ax.annotate(
            r["combo_pair"],
            (r["vital_organ_max_bliss"], r["rd_patient_synergy_bliss"]),
            fontsize=7.5,
            xytext=(5, 3),
            textcoords="offset points",
            fontweight="bold",
        )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Max Vital Normal Organ Synergy (GTEx Off-Target Risk)", fontsize=10)
    ax.set_ylabel("Mean Chemoresistant RD Patient Tumor Synergy (Bliss)", fontsize=10)
    ax.set_title("Exp23: Pareto Discovery of Golden-Window Rescue Combos\n(1,770 Drug Pairs Screened on 80 Chemoresistant Patients vs 8 GTEx Vital Organs)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp23 results to {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
