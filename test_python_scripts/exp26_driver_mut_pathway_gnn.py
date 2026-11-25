import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gseapy as gp
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

TCGA_RNA_DIR = r"D:\data\tcga\rnaseq"
TCGA_RPPA_PATH = r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena"
TCGA_MAF_DIR = r"D:\TCGA\TCGA\mutation_data"
HOTSPOT_PATH = r"D:\data\drug\DeepMapcrisp\OmicsSomaticMutationsMatrixHotspot.csv"
CACHE_24 = r"D:\data\AIVC\data\exp24_multiomics_cache.npz"

OUT_CSV = r"D:\data\AIVC\test_results\exp26_driver_mut_reggnn_results.csv"
OUT_TARGET_CSV = r"D:\data\AIVC\test_results\exp26_driver_mut_key_proteins.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp26_driver_mut_reggnn_plot.png"

UNSEEN_CANCERS = {"LUAD", "OV"}

RPPA_TO_GENE = {
    "AKT": "AKT1", "AKT_pS473": "AKT1", "AKT_pT308": "AKT1",
    "MAPK_pT202Y204": "MAPK1", "ERK2": "MAPK1", "MEK1": "MAP2K1", "MEK1_pS217S221": "MAP2K1",
    "EGFR": "EGFR", "EGFR_pY1068": "EGFR", "EGFR_pY1173": "EGFR",
    "HER2": "ERBB2", "HER2_pY1248": "ERBB2", "HER3": "ERBB3",
    "X4EBP1": "EIF4EBP1", "X4EBP1_pS65": "EIF4EBP1", "X4EBP1_pT37T46": "EIF4EBP1",
    "S6_pS235S236": "RPS6", "S6_pS240S244": "RPS6", "P70S6K1": "RPS6KB1", "P70S6K_pT389": "RPS6KB1",
    "MTOR": "MTOR", "MTOR_pS2448": "MTOR", "AMPKALPHA": "PRKAA1", "AMPKALPHA_pT172": "PRKAA1",
    "STAT3_pY705": "STAT3", "STAT5ALPHA": "STAT5A", "JAK2": "JAK2",
    "RB": "RB1", "RB_pS807S811": "RB1", "CYCLIND1": "CCND1", "CYCLINB1": "CCNB1", "CYCLINE1": "CCNE1",
    "CDK1": "CDK1", "P27": "CDKN1B", "P21": "CDKN1A", "TP53": "TP53", "P53": "TP53",
    "BCL2": "BCL2", "BCLXL": "BCL2L1", "BAX": "BAX", "CASPASE7CLEAVEDD198": "CASP7",
    "PARP1": "PARP1", "ATM": "ATM", "CHEK1": "CHEK1", "CHEK2": "CHEK2",
    "PI3KP110ALPHA": "PIK3CA", "PTEN": "PTEN", "SRC": "SRC", "SRC_pY527": "SRC",
    "CMET": "MET", "CMET_pY1235": "MET", "BRAF": "BRAF", "CRAF": "RAF1",
    "BETACATENIN": "CTNNB1", "ECADHERIN": "CDH1", "NCADHERIN": "CDH2", "SNAI1": "SNAI1",
    "YAP": "YAP1", "YAP_pS127": "YAP1", "NFKBP65_pS536": "RELA",
}

KEY_MUT_HUBS = ["P53", "PTEN", "RB", "AKT_pS473", "AKT_pT308", "MAPK_pT202Y204", "MEK1_pS217S221", "EGFR_pY1068", "HER2_pY1248", "BRAF", "PI3KP110ALPHA", "X4EBP1_pS65", "S6_pS235S236"]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_kegg_graph(rna_genes: list, max_pws: int = 45) -> np.ndarray:
    kegg = gp.get_library("KEGG_2021_Human")
    gset = set(rna_genes)
    g2i = {g: i for i, g in enumerate(rna_genes)}
    kw = ("signaling", "cancer", "cell cycle", "apoptosis", "p53", "dna replication", "proteasome", "glycolysis")
    cand = []
    for pn, pgenes in kegg.items():
        ov = [g.upper() for g in pgenes if g.upper() in gset]
        if len(ov) >= 12:
            sc = (1000 if any(k in pn.lower() for k in kw) else 0) + len(ov)
            cand.append((sc, pn, ov))
    cand.sort(reverse=True)
    sel = cand[:max_pws]
    A = np.zeros((len(rna_genes), len(sel)), dtype=np.float32)
    for j, (_, _, ov) in enumerate(sel):
        for g in ov:
            A[g2i[g], j] = 1.0
    return (A / np.sqrt(np.maximum(A.sum(axis=0, keepdims=True), 1.0))).astype(np.float32)


class MutAwarePathwayGNN(nn.Module):
    """Pathway-GNN with optional 165 Hotspot Driver Mutation Gate (Modality Dropout)."""
    def __init__(self, n_rna: int, n_mut: int, rppa_cis_idx: list, A_gp: np.ndarray, use_mut: bool = True, hid: int = 192):
        super().__init__()
        self.use_mut = use_mut
        self.register_buffer("A_gp", torch.tensor(A_gp, dtype=torch.float32))
        self.register_buffer("rppa_cis_idx", torch.tensor(rppa_cis_idx, dtype=torch.long))
        n_rppa = len(rppa_cis_idx)
        self.rppa_cis_w = nn.Parameter(torch.ones(n_rppa) * 0.35)

        self.pw_enc = nn.Sequential(nn.Linear(A_gp.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.glob_enc = nn.Sequential(nn.Linear(n_rna, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.3))
        self.mut_enc = nn.Sequential(nn.Linear(n_mut, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2))

        in_d = hid * 2 + (96 if use_mut else 0)
        self.fuse = nn.Sequential(nn.Linear(in_d, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.dec = nn.Linear(hid, n_rppa)
        nn.init.zeros_(self.dec.weight)

    def forward(self, x_rna: torch.Tensor, x_mut: torch.Tensor = None) -> torch.Tensor:
        h_pw = self.pw_enc(x_rna @ self.A_gp)
        h_gl = self.glob_enc(x_rna)
        if self.use_mut:
            hm = self.mut_enc(x_mut) if x_mut is not None else torch.zeros((x_rna.shape[0], 96), device=x_rna.device, dtype=x_rna.dtype)
            h = self.fuse(torch.cat([h_pw, h_gl, hm], dim=-1))
        else:
            h = self.fuse(torch.cat([h_pw, h_gl], dim=-1))
        vm = (self.rppa_cis_idx >= 0).float()
        si = torch.clamp(self.rppa_cis_idx, min=0)
        return x_rna[:, si] * vm * self.rppa_cis_w + 0.45 * self.dec(h)


def main():
    print(f"=== Exp26: TCGA 165 Hotspot Driver Mutations + RNA Pathway-GNN ({DEVICE}) ===")

    # 1. Load 165 Hotspot Driver Gene List
    hot = pd.read_csv(HOTSPOT_PATH, nrows=1)
    driver_genes = [str(c).split(" (")[0].strip().upper() for c in hot.columns[6:]]
    driver_set = set(driver_genes)
    drv_pos = {g: i for i, g in enumerate(driver_genes)}
    print(f"Loaded {len(driver_genes)} DepMap Hotspot Driver Genes.")

    # 2. Load top_rna & rppa_abs from exp24 cache, and align TCGA RNA + RPPA + MAF sample IDs
    c24 = np.load(CACHE_24, allow_pickle=True)
    top_rna = [str(g) for g in c24["top_rna"]]
    rna_pos = {g: i for i, g in enumerate(top_rna)}

    rppa = pd.read_csv(TCGA_RPPA_PATH, sep="\t", index_col=0)
    rppa = rppa[rppa.isna().mean(axis=1) < 0.30]
    rppa_abs = [str(a).strip() for a in rppa.index]
    rppa_sids = set(rppa.columns)

    # Parse all TCGA_*_mutect2_maf files (non-silent coding mutations in 165 driver genes)
    print("Parsing 32 TCGA Mutect2 MAF files for 165 Hotspot Driver Mutations...")
    sid_to_mut = {}
    maf_files = sorted(glob.glob(os.path.join(TCGA_MAF_DIR, "TCGA_*_mutect2_maf")))
    for mf in maf_files:
        df_m = pd.read_csv(
            mf, sep="\t", comment="#",
            usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"],
            low_memory=False
        )
        df_m["Hugo_Symbol"] = df_m["Hugo_Symbol"].astype(str).str.strip().str.upper()
        df_m = df_m[(df_m["Variant_Classification"] != "Silent") & (df_m["Hugo_Symbol"].isin(driver_set))]
        for sym, bc in zip(df_m["Hugo_Symbol"].values, df_m["Tumor_Sample_Barcode"].astype(str).values):
            s15 = bc[:15]
            if s15 in rppa_sids:
                if s15 not in sid_to_mut:
                    sid_to_mut[s15] = np.zeros(len(driver_genes), dtype=np.float32)
                sid_to_mut[s15][drv_pos[sym]] = 1.0
    print(f"  Matched TCGA RPPA patients with >=1 non-silent driver mutation: {len(sid_to_mut)}")

    # Load TCGA RNA + RPPA + Driver Mutations in exact aligned order
    x_list, m_list, y_list, c_list = [], [], [], []
    tpm_files = sorted(glob.glob(os.path.join(TCGA_RNA_DIR, "*_mrna_tpm.csv")))
    for fp in tpm_files:
        ctype = os.path.basename(fp).replace("TCGA-", "").replace("_mrna_tpm.csv", "")
        df_r = pd.read_csv(fp, low_memory=False)
        df_r["gene"] = df_r["gene"].astype(str).str.strip().str.upper()
        df_r = df_r[df_r["gene"].isin(top_rna)].drop_duplicates("gene").set_index("gene")
        col_to_sid15 = {c: str(c)[:15] for c in df_r.columns if str(c)[:15] in rppa_sids}
        if not col_to_sid15:
            continue
        seen_s15 = {}
        for orig_c, s15 in col_to_sid15.items():
            if s15 not in seen_s15:
                seen_s15[s15] = orig_c
        s15_list = sorted(seen_s15.keys())
        orig_cols = [seen_s15[s] for s in s15_list]

        rna_mat = df_r[orig_cols].reindex(top_rna).fillna(0.0).values.T
        rna_z = row_zscore(np.log2(np.clip(rna_mat, 0.0, None) + 1.0))

        y_rppa = rppa[s15_list].values.T.astype(np.float32)
        y_rppa = np.nan_to_num(y_rppa, nan=0.0, posinf=0.0, neginf=0.0)

        mut_mat = np.stack([sid_to_mut.get(s, np.zeros(len(driver_genes), dtype=np.float32)) for s in s15_list], axis=0)
        x_list.append(rna_z)
        m_list.append(mut_mat)
        y_list.append(y_rppa)
        c_list.extend([ctype] * len(s15_list))

    X_rna = np.vstack(x_list)
    X_mut = np.vstack(m_list)
    Y_rppa = np.vstack(y_list)
    C_tcga = np.array(c_list)
    print(f"Total TCGA 3-Omics Paired Patients (RNA + 165 DriverMut + RPPA): {len(X_rna)}")

    tr_mask = ~np.isin(C_tcga, list(UNSEEN_CANCERS))
    te_mask = np.isin(C_tcga, list(UNSEEN_CANCERS))

    rppa_cis_idx = [rna_pos.get(RPPA_TO_GENE.get(ab, ab.split("_")[0].lstrip("X").upper()), -1) for ab in rppa_abs]
    A_gp = build_kegg_graph(top_rna, max_pws=45)

    def train_eval(use_mut: bool):
        torch.manual_seed(SEED)
        net = MutAwarePathwayGNN(len(top_rna), len(driver_genes), rppa_cis_idx, A_gp, use_mut=use_mut).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.tensor(X_rna[tr_mask]), torch.tensor(X_mut[tr_mask]), torch.tensor(Y_rppa[tr_mask])), batch_size=256, shuffle=True)
        net.train()
        for _ in range(35):
            for br, bm, by in loader:
                br, bm, by = br.to(DEVICE), bm.to(DEVICE), by.to(DEVICE)
                bm_in = None if (use_mut and np.random.rand() < 0.25) else bm
                opt.zero_grad()
                loss = loss_fn(net(br, bm_in), by)
                loss.backward()
                opt.step()
        net.eval()
        return net

    net_nomut = train_eval(use_mut=False)
    net_mut = train_eval(use_mut=True)

    summary_rows = []
    key_rows = []
    with torch.no_grad():
        for c in sorted(UNSEEN_CANCERS):
            m = (C_tcga == c)
            xr = torch.tensor(X_rna[m], device=DEVICE)
            xm = torch.tensor(X_mut[m], device=DEVICE)
            yt = Y_rppa[m]

            p_nomut = net_nomut(xr, None).cpu().numpy()
            p_mut = net_mut(xr, xm).cpu().numpy()
            p_fb = net_mut(xr, None).cpu().numpy()

            for mname, yp in [("1_RNA_Only_PathwayGNN", p_nomut), ("2_RNA+165DriverMut_PathwayGNN", p_mut), ("3_MutOmitted_Fallback", p_fb)]:
                all_r = [pearsonr(yp[:, j], yt[:, j])[0] for j in range(len(rppa_abs)) if np.std(yt[:, j]) > 1e-6]
                phos_r = [pearsonr(yp[:, j], yt[:, j])[0] for j, ab in enumerate(rppa_abs) if ("_p" in ab and np.std(yt[:, j]) > 1e-6)]
                med_all, med_phos = float(np.median(all_r)), float(np.median(phos_r))
                print(f"  [BLIND TCGA-{c:4s}] {mname:30s} | All RPPA Med r={med_all:.4f} | Phospho Med r={med_phos:.4f}")
                summary_rows.append({"cancer": c, "model": mname, "all_rppa_med_r": med_all, "phospho_med_r": med_phos})

            # Key mutation-driven hubs comparison
            for ab in KEY_MUT_HUBS:
                if ab in rppa_abs:
                    j = rppa_abs.index(ab)
                    if np.std(yt[:, j]) > 1e-6:
                        r0 = pearsonr(p_nomut[:, j], yt[:, j])[0]
                        r1 = pearsonr(p_mut[:, j], yt[:, j])[0]
                        key_rows.append({"cancer": c, "antibody": ab, "rna_only_r": r0, "rna_plus_mut_r": r1, "mut_gain": r1 - r0})

    sum_df = pd.DataFrame(summary_rows)
    key_df = pd.DataFrame(key_rows).sort_values(["cancer", "mut_gain"], ascending=[True, False])
    sum_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    key_df.to_csv(OUT_TARGET_CSV, index=False, encoding="utf-8-sig")
    print("\n=== Impact of 165 Driver Mutations on Key Signaling & Tumor-Suppressor Hubs ===")
    print(key_df.to_string(index=False))

    # Plot 2-panel summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    piv = sum_df.pivot(index="cancer", columns="model", values="all_rppa_med_r")
    piv.plot(kind="bar", ax=axes[0], width=0.72, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Blind Unseen Cancers (LUAD & OV): RPPA Protein Prediction\nRNA-Only vs RNA + 165 Driver Mutations vs Fallback", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Median Cross-Patient Pearson (r)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8)

    luad_k = key_df[key_df["cancer"] == "LUAD"].head(10)
    yp = np.arange(len(luad_k))
    axes[1].barh(yp - 0.18, luad_k["rna_only_r"], height=0.35, label="RNA Only", color="#95a5a6", edgecolor="black", linewidth=0.5)
    axes[1].barh(yp + 0.18, luad_k["rna_plus_mut_r"], height=0.35, label="RNA + 165 Driver Mutations", color="#8e44ad", edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(yp)
    axes[1].set_yticklabels(luad_k["antibody"], fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cross-Patient Pearson (r) in Unseen LUAD")
    axes[1].set_title("Driver Mutation Boost on Key Hubs\n(P53, AKT/MAPK Phospho, PTEN, RB, EGFR)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp26 results to {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
