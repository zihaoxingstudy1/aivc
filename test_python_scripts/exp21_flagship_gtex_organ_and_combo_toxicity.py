import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_SINGLE_CSV = r"D:\data\AIVC\test_results\exp21_single_drug_gtex_toxicity.csv"
OUT_COMBO_CSV = r"D:\data\AIVC\test_results\exp21_combo_gtex_synergy_toxicity.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp21_gtex_organ_and_combo_toxicity.png"

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

TOXIC_CHEMO_BENCHMARK = [
    "Bortezomib", "Gemcitabine", "Doxorubicin", "Epirubicin",
    "Paclitaxel", "Docetaxel", "Cisplatin", "Oxaliplatin",
    "Cyclophosphamide", "5-Fluorouracil", "Methotrexate", "Vincristine",
]

TARGETED_SAFE_CONTROLS = [
    "Dabrafenib", "Erlotinib", "Gefitinib", "Osimertinib", "Lapatinib", "Imatinib",
]

COMBO_PAIRS_TO_AUDIT = [
    ("Paclitaxel", "Doxorubicin"),
    ("Doxorubicin", "Cyclophosphamide"),
    ("5-Fluorouracil", "Oxaliplatin"),
    ("Gemcitabine", "Cisplatin"),
    ("Bortezomib", "Vorinostat"),
    ("Dabrafenib", "Trametinib"),
    ("Osimertinib", "Gefitinib"),
    ("Paclitaxel", "Cisplatin"),
    ("Irinotecan", "5-Fluorouracil"),
    ("Epirubicin", "Cyclophosphamide"),
]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class BioMolKGIC50Net(nn.Module):
    def __init__(self, n_genes: int, n_drug: int = 2048, hid: int = 256):
        super().__init__()
        self.c_enc = nn.Sequential(nn.Linear(n_genes, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.d_enc = nn.Sequential(nn.Linear(n_drug, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, xc, xd):
        return self.head(torch.cat([self.c_enc(xc), self.d_enc(xd)], dim=-1)).squeeze(-1)


class BioMolKGSynergyNet(nn.Module):
    """Symmetric permutation-invariant dual-drug + cell/organ synergy predictor."""
    def __init__(self, n_genes: int, n_drug: int = 2048, hid: int = 256):
        super().__init__()
        self.c_enc = nn.Sequential(nn.Linear(n_genes, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.d_enc = nn.Sequential(nn.Linear(n_drug, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, 128), nn.LayerNorm(128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128 * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, xc, xd1, xd2):
        hc = self.c_enc(xc)
        hd1 = self.d_enc(xd1)
        hd2 = self.d_enc(xd2)
        sym_sum = hd1 + hd2
        sym_prod = hd1 * hd2
        return self.head(torch.cat([hc, sym_sum, sym_prod], dim=-1)).squeeze(-1)


def main():
    print(f"=== Exp21: Flagship GTEx 54-Organ Single & Combo Toxicity Benchmark ({DEVICE}) ===")

    # 1. Load GTEx v8 54 normal organs & Cell Line RNA-seq
    print("Loading GTEx v8 54 normal organs & Cell Line RNA-seq...")
    gtex_df = pd.read_csv(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", sep="\t", skiprows=2)
    gtex_df["Description"] = gtex_df["Description"].astype(str).str.strip().str.upper()
    organ_cols = [c for c in gtex_df.columns if c not in ("Name", "Description")]
    gtex_gene = gtex_df.groupby("Description", sort=False)[organ_cols].mean()
    gtex_log2 = np.log2(np.clip(gtex_gene.values, 0.0, None) + 1.0)
    gtex_log2_df = pd.DataFrame(gtex_log2.T, index=organ_cols, columns=gtex_gene.index)  # 54 organs x genes

    rna_raw = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", skiprows=[1, 2], low_memory=False)
    rna_raw = rna_raw.rename(columns={rna_raw.columns[0]: "gene_symbol"})
    rna_raw = rna_raw.drop(columns=[c for c in rna_raw.columns if str(c).startswith("Unnamed:")])
    rna_raw["gene_symbol"] = rna_raw["gene_symbol"].astype(str).str.strip().str.upper()
    shared_genes = sorted(set(rna_raw["gene_symbol"]) & set(gtex_log2_df.columns))
    rna_raw = rna_raw[rna_raw["gene_symbol"].isin(shared_genes)].drop_duplicates(subset=["gene_symbol"]).set_index("gene_symbol")
    rna_raw = rna_raw.apply(pd.to_numeric, errors="coerce")

    top_2000 = sorted(list(rna_raw.var(axis=1).sort_values(ascending=False).index[:2000]))
    print(f"Shared genes: {len(shared_genes)} | Selected top_2000: {len(top_2000)}")

    cell_expr_df = rna_raw.loc[top_2000].T.dropna(how="all")
    cell_z = row_zscore(cell_expr_df.values)
    cell_id_to_row = {sid: i for i, sid in enumerate(cell_expr_df.index)}

    gtex_z = row_zscore(gtex_log2_df[top_2000].values)
    organ_to_row = {org: i for i, org in enumerate(organ_cols)}

    # 2. Train Latest Pan-Cancer BioMolKG-IC50 on ALL GDSC1 + GDSC2 (preferring rich 3,142-drug combo_drug_kg_features over gdsc fallback)
    c_npz = np.load(r"D:\data\AIVC\data\combo_drug_kg_features.npz", allow_pickle=True)
    combo_d2f = {str(d).upper(): c_npz["feats"][i].astype(np.float32) for i, d in enumerate(c_npz["drugs"])}
    gdsc_npz = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=True)
    gdsc_d2f = {str(d).upper(): gdsc_npz["feats"][i].astype(np.float32) for i, d in enumerate(gdsc_npz["drugs"])}
    gdsc_d2name = {str(d).upper(): str(d) for d in gdsc_npz["drugs"]}
    for du, df in combo_d2f.items():
        gdsc_d2f[du] = df
        if du not in gdsc_d2name:
            gdsc_d2name[du] = du.title()

    g1 = pd.read_excel(r"D:\data\AIVC\data\GDSC1_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    gdsc = pd.concat([g1, g2], ignore_index=True).dropna()
    gdsc["du"] = gdsc["DRUG_NAME"].astype(str).str.strip().str.upper()
    gdsc = gdsc[gdsc["SANGER_MODEL_ID"].isin(cell_id_to_row) & gdsc["du"].isin(gdsc_d2f)].copy()

    c_idx = gdsc["SANGER_MODEL_ID"].map(cell_id_to_row).values
    uniq_d, d_inv = np.unique(gdsc["du"].values, return_inverse=True)
    d_mat = np.stack([gdsc_d2f[u] for u in uniq_d], axis=0)
    y_ic50 = gdsc["LN_IC50"].values.astype(np.float32)
    print(f"Training Pan-Cancer BioMolKG-IC50 on {len(y_ic50)} GDSC1+2 pairs ({len(uniq_d)} drugs)...")

    ic50_net = BioMolKGIC50Net(len(top_2000), 2048).to(DEVICE)
    opt = torch.optim.AdamW(ic50_net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    cell_z_t = torch.tensor(cell_z, device=DEVICE)
    d_mat_t = torch.tensor(d_mat, device=DEVICE)
    loader = DataLoader(TensorDataset(torch.tensor(c_idx), torch.tensor(d_inv), torch.tensor(y_ic50)), batch_size=2048, shuffle=True)
    ic50_net.train()
    for ep in range(6):
        tot = 0.0
        for bc, bd, by in loader:
            opt.zero_grad()
            loss = loss_fn(ic50_net(cell_z_t[bc], d_mat_t[bd]), by.to(DEVICE))
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(by)
        print(f"  IC50 Epoch {ep+1}/6 | MSE={tot / len(y_ic50):.4f}")

    # Evaluate GTEx 54-Organ Toxicity & Benchmark vs Clinical Ground Truth
    ic50_net.eval()
    gtex_t = torch.tensor(gtex_z, device=DEVICE)
    all_bench_drugs = TOXIC_CHEMO_BENCHMARK + TARGETED_SAFE_CONTROLS
    single_rows = []
    with torch.no_grad():
        for dname in all_bench_drugs:
            du = dname.upper()
            if du not in gdsc_d2f:
                print(f"  [MISS] {dname} not in GDSC drugs")
                continue
            dfeat = torch.tensor(gdsc_d2f[du], device=DEVICE).unsqueeze(0).expand(len(organ_cols), -1)
            pred_ln_ic50 = ic50_net(gtex_t, dfeat).cpu().numpy()
            tox_class = "Cytotoxic_OrganToxic" if dname in TOXIC_CHEMO_BENCHMARK else "Targeted_LowTox"
            row = {"drug": gdsc_d2name[du], "clinical_class": tox_class, "mean_54organ_LN_IC50": float(np.mean(pred_ln_ic50))}
            for org in VITAL_ORGANS:
                if org in organ_to_row:
                    row[org] = float(pred_ln_ic50[organ_to_row[org]])
            single_rows.append(row)

    single_df = pd.DataFrame(single_rows).sort_values("mean_54organ_LN_IC50")
    single_df.to_csv(OUT_SINGLE_CSV, index=False, encoding="utf-8-sig")

    # Statistical Verification vs Clinical Ground Truth (Lower LN_IC50 = Higher Normal Organ Toxicity)
    y_true_tox = (single_df["clinical_class"] == "Cytotoxic_OrganToxic").astype(int).values
    y_score_tox = -single_df["mean_54organ_LN_IC50"].values
    tox_auc = roc_auc_score(y_true_tox, y_score_tox)
    g_tox = single_df[single_df["clinical_class"] == "Cytotoxic_OrganToxic"]["mean_54organ_LN_IC50"].values
    g_safe = single_df[single_df["clinical_class"] == "Targeted_LowTox"]["mean_54organ_LN_IC50"].values
    _, p_mw = mannwhitneyu(g_tox, g_safe, alternative="less")
    print(f"\n=== Single-Drug GTEx Normal Organ Toxicity Benchmark vs Clinical Labels ===")
    print(f"  Cytotoxic Mean LN_IC50={np.mean(g_tox):.3f} vs Targeted Control Mean={np.mean(g_safe):.3f}")
    print(f"  Clinical Toxicity Separation ROC-AUC = {tox_auc:.4f} (Mann-Whitney p = {p_mw:.2e})")
    print(single_df[["drug", "clinical_class", "mean_54organ_LN_IC50", "Whole Blood", "Liver", "Heart - Left Ventricle", "Kidney - Cortex"]].to_string(index=False))

    # 3. Train Latest BioMolKG-Synergy on DrugComb v1.5 (342k combos) & Predict Tumor vs Vital Organ Combo Synergy
    print("\n=== Training BioMolKG-Synergy on DrugComb v1.5 & Predicting Organ Combo Toxicity ===")

    cmp = pd.read_csv(r"D:\data\AIVC\data\cmp_model_list.csv", usecols=["model_id", "model_name"]).dropna()
    cname_to_row = {}
    for mid, mname in zip(cmp["model_id"], cmp["model_name"]):
        if mid in cell_id_to_row:
            cname_to_row[str(mname).strip().upper()] = cell_id_to_row[mid]

    dc = pd.read_csv(
        r"D:\data\drug\summary_v_1_5.csv",
        usecols=["drug_row", "drug_col", "cell_line_name", "synergy_bliss"],
        low_memory=False
    ).dropna(subset=["drug_row", "drug_col", "cell_line_name", "synergy_bliss"])
    dc["d1u"] = dc["drug_row"].astype(str).str.strip().str.upper()
    dc["d2u"] = dc["drug_col"].astype(str).str.strip().str.upper()
    dc["cu"] = dc["cell_line_name"].astype(str).str.strip().str.upper()
    dc = dc[dc["d1u"].isin(combo_d2f) & dc["d2u"].isin(combo_d2f) & dc["cu"].isin(cname_to_row)].copy()
    dc["synergy_bliss"] = pd.to_numeric(dc["synergy_bliss"], errors="coerce")
    dc = dc.dropna(subset=["synergy_bliss"])
    # Clip extreme outlier bliss scores for stable regression
    dc["synergy_bliss"] = np.clip(dc["synergy_bliss"].values, -50.0, 50.0).astype(np.float32)

    sc_idx = dc["cu"].map(cname_to_row).values
    all_u = sorted(set(dc["d1u"]) | set(dc["d2u"]))
    u2i = {u: i for i, u in enumerate(all_u)}
    sd1_idx = dc["d1u"].map(u2i).values
    sd2_idx = dc["d2u"].map(u2i).values
    sd_mat = np.stack([combo_d2f[u] for u in all_u], axis=0)
    y_bliss = dc["synergy_bliss"].values
    print(f"Training BioMolKG-Synergy on {len(y_bliss)} DrugComb v1.5 combinations ({len(all_u)} drugs)...")

    syn_net = BioMolKGSynergyNet(len(top_2000), 2048).to(DEVICE)
    opt_s = torch.optim.AdamW(syn_net.parameters(), lr=1e-3, weight_decay=1e-4)
    sd_mat_t = torch.tensor(sd_mat, device=DEVICE)
    s_loader = DataLoader(TensorDataset(torch.tensor(sc_idx), torch.tensor(sd1_idx), torch.tensor(sd2_idx), torch.tensor(y_bliss)), batch_size=2048, shuffle=True)
    syn_net.train()
    for ep in range(6):
        tot = 0.0
        for bc, bd1, bd2, by in s_loader:
            opt_s.zero_grad()
            loss = loss_fn(syn_net(cell_z_t[bc], sd_mat_t[bd1], sd_mat_t[bd2]), by.to(DEVICE))
            loss.backward()
            opt_s.step()
            tot += float(loss.item()) * len(by)
        print(f"  Synergy Epoch {ep+1}/6 | MSE={tot / len(y_bliss):.4f}")

    # Predict Combo Synergy on Representative Tumor Lines (MCF7, HCT116, K562) vs Vital GTEx Organs
    syn_net.eval()
    tumor_lines = [cl for cl in ["MCF7", "HCT116", "K562"] if cl in cname_to_row]
    tumor_z_t = torch.tensor(np.stack([cell_z[cname_to_row[cl]] for cl in tumor_lines]), device=DEVICE)
    vital_idx = [organ_to_row[o] for o in VITAL_ORGANS if o in organ_to_row]
    vital_names = [o for o in VITAL_ORGANS if o in organ_to_row]
    vital_z_t = torch.tensor(gtex_z[vital_idx], device=DEVICE)

    combo_rows = []
    with torch.no_grad():
        for d1, d2 in COMBO_PAIRS_TO_AUDIT:
            d1u, d2u = d1.upper(), d2.upper()
            if d1u not in combo_d2f or d2u not in combo_d2f:
                print(f"  [MISS COMBO] {d1} + {d2}")
                continue
            f1 = torch.tensor(combo_d2f[d1u], device=DEVICE).unsqueeze(0)
            f2 = torch.tensor(combo_d2f[d2u], device=DEVICE).unsqueeze(0)
            t_syn = syn_net(tumor_z_t, f1.expand(len(tumor_lines), -1), f2.expand(len(tumor_lines), -1)).cpu().numpy()
            o_syn = syn_net(vital_z_t, f1.expand(len(vital_names), -1), f2.expand(len(vital_names), -1)).cpu().numpy()

            tumor_mean = float(np.mean(t_syn))
            organ_max = float(np.max(o_syn))
            organ_mean = float(np.mean(o_syn))
            row = {
                "combo_pair": f"{d1} + {d2}",
                "tumor_mean_bliss": tumor_mean,
                "vital_organ_mean_bliss": organ_mean,
                "vital_organ_max_bliss": organ_max,
                "therapeutic_window_index": tumor_mean - organ_max,
            }
            for oname, oval in zip(vital_names, o_syn):
                row[f"Organ_{oname}"] = float(oval)
            combo_rows.append(row)

    combo_df = pd.DataFrame(combo_rows).sort_values("therapeutic_window_index", ascending=False)
    combo_df.to_csv(OUT_COMBO_CSV, index=False, encoding="utf-8-sig")
    print("\n=== Combo Synergy on Tumor vs Vital Normal Organs (Therapeutic Window) ===")
    print(combo_df[["combo_pair", "tumor_mean_bliss", "vital_organ_mean_bliss", "vital_organ_max_bliss", "therapeutic_window_index"]].to_string(index=False))

    # 4. Plot 2-Panel Summary Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), dpi=180)
    # Panel A: Single-drug GTEx organ toxicity heatmap
    heat_cols = ["Whole Blood", "Liver", "Heart - Left Ventricle", "Kidney - Cortex", "Lung", "Brain - Cortex"]
    hmat = single_df.set_index("drug")[heat_cols]
    im = axes[0].imshow(hmat.values, cmap="RdYlBu", aspect="auto", vmin=-4.5, vmax=4.5)
    axes[0].set_yticks(np.arange(len(hmat)))
    axes[0].set_yticklabels([f"{d} ({'Tox' if c=='Cytotoxic_OrganToxic' else 'Safe'})" for d, c in zip(hmat.index, single_df["clinical_class"])], fontsize=8)
    axes[0].set_xticks(np.arange(len(heat_cols)))
    axes[0].set_xticklabels([c.replace("_", "\n") for c in heat_cols], fontsize=8)
    axes[0].set_title(f"GTEx Normal Organ Toxicity (Predicted LN_IC50)\nChemo vs Targeted ROC-AUC = {tox_auc:.3f} (p={p_mw:.1e})", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Predicted LN_IC50 (Red=Toxic, Blue=Safe)")

    # Panel B: Combo Tumor Synergy vs Max Vital Organ Synergy
    y_pos = np.arange(len(combo_df))
    axes[1].barh(y_pos - 0.18, combo_df["tumor_mean_bliss"], height=0.35, label="Tumor Cell Mean Synergy (Bliss)", color="#2ecc71", edgecolor="black", linewidth=0.5)
    axes[1].barh(y_pos + 0.18, combo_df["vital_organ_max_bliss"], height=0.35, label="Vital Organ Max Synergy (Off-Target Risk)", color="#e74c3c", edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(combo_df["combo_pair"], fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Predicted Bliss Synergy Score")
    axes[1].set_title("Multi-Drug Combo Therapeutic Window\n(Tumor Efficacy Synergy vs Vital Organ Toxicity Synergy)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp21 figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
