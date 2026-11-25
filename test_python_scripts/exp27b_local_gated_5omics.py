import os
import re
import sys
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

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TCGA_RNA_DIR = r"D:\data\tcga\rnaseq"
TCGA_RPPA_PATH = r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena"
TCGA_MAF_DIR = r"D:\TCGA\TCGA\mutation_data"
CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
CPTAC_MUT_DIR = r"D:\data\CPTAC\Mutation_BCM_v1"
CCLE_RPPA_PATH = r"D:\data\tcga\protein\CCLE_RPPA_20180123.csv"
CCLE_TMT_PATH = r"D:\data\tcga\protein\ccle-protein_quant_current_normalized.csv.gz"
HOTSPOT_PATH = r"D:\data\drug\DeepMapcrisp\OmicsSomaticMutationsMatrixHotspot.csv"
BIOGRID_PATH = r"D:\data\regulation\BIOGRID-ALL-4.4.211.tab3.txt"
CELL_RNA_PATH = r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv"
CMP_PATH = r"D:\data\AIVC\data\cmp_model_list.csv"
CACHE_24 = r"D:\data\AIVC\data\exp24_multiomics_cache.npz"

OUT_CKPT = r"D:\data\AIVC\pretrained\aivc_5omics_pathway_gnn.pt"
OUT_CSV = r"D:\data\AIVC\test_results\exp27b_local_gated_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp27b_local_gated_comparison.png"

UNSEEN_CANCERS = {"LUAD", "OV"}

CPTAC_MUT_PREFIX = {
    "brca": "BRCA", "ccrcc": "ccRCC", "coad": "COAD", "gbm": "GBM",
    "hnscc": "HNSCC", "lscc": "LSCC", "luad": "LUAD", "ov": "HGSC",
    "pdac": "PDAC", "ucec": "UCEC",
}

CORE_TFS = [
    "TP53", "MYC", "E2F1", "E2F3", "HIF1A", "STAT3", "STAT1", "ESR1", "FOXA1", "FOXM1",
    "JUN", "FOS", "RELA", "NFKB1", "SP1", "ETS1", "CTNNB1", "SMAD3", "SMAD4", "AR",
    "SOX2", "KLF4", "BRCA1", "YAP1", "TEAD1", "CEBPB", "ATF4", "NFE2L2", "CREB1",
    "TWIST1", "SNAI1", "ZEB1", "REST", "YY1", "MAX", "EGR1", "IRF1", "GATA3", "RUNX1",
    "PAX8", "NKX2-1", "HNF4A", "PPARG", "RXRA", "SRF", "ELK1", "MITF", "FOXO1", "FOXO3",
]

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


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def norm_cell_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def norm_ab_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper().lstrip("X"))


def build_graphs_and_mut_masks(rna_genes: list, driver_genes: list, rppa_target_genes: list, prot_genes: list, phos_genes: list):
    """Build BioGRID TF graph, KEGG Pathway graph, AND strict 1-hop Driver->Target mutation masks."""
    gset = set(rna_genes)
    g2i = {g: i for i, g in enumerate(rna_genes)}
    drv_set = set(driver_genes)

    tf_list = [tf for tf in CORE_TFS if tf in gset]
    tf_set = set(tf_list)
    tf2i = {tf: j for j, tf in enumerate(tf_list)}
    A_tf = np.zeros((len(rna_genes), len(tf_list)), dtype=np.float32)

    # Track 1-hop BioGRID neighbors of each driver gene
    drv_neighbors = {d: {d} for d in driver_genes}
    print("Building BioGRID TF-Gene, Driver 1-hop Neighbors & KEGG Pathway Graphs...")
    for chunk in pd.read_csv(
        BIOGRID_PATH, sep="\t",
        usecols=["Official Symbol Interactor A", "Official Symbol Interactor B", "Organism ID Interactor A", "Organism ID Interactor B"],
        chunksize=300000, low_memory=False
    ):
        hu = chunk[(chunk["Organism ID Interactor A"] == 9606) & (chunk["Organism ID Interactor B"] == 9606)]
        for ga, gb in zip(hu["Official Symbol Interactor A"].astype(str).str.upper(), hu["Official Symbol Interactor B"].astype(str).str.upper()):
            if ga in tf_set and gb in gset:
                A_tf[g2i[gb], tf2i[ga]] = 1.0
            if gb in tf_set and ga in gset:
                A_tf[g2i[ga], tf2i[gb]] = 1.0
            if ga in drv_set:
                drv_neighbors[ga].add(gb)
            if gb in drv_set:
                drv_neighbors[gb].add(ga)

    for tf in tf_list:
        A_tf[g2i[tf], tf2i[tf]] = 1.0
    A_tf_norm = (A_tf / np.sqrt(np.maximum(A_tf.sum(axis=0, keepdims=True), 1.0))).astype(np.float32)

    kegg = gp.get_library("KEGG_2021_Human")
    kw = ("signaling", "cancer", "cell cycle", "apoptosis", "p53", "dna replication", "proteasome", "glycolysis")
    cand = []
    for pn, pg in kegg.items():
        ov = [g.upper() for g in pg if g.upper() in gset]
        if len(ov) >= 12:
            cand.append(((1000 if any(k in pn.lower() for k in kw) else 0) + len(ov), pn, ov))
    cand.sort(reverse=True)
    sel = cand[:45]
    A_gp = np.zeros((len(rna_genes), len(sel)), dtype=np.float32)
    for j, (_, _, ov) in enumerate(sel):
        for g in ov:
            A_gp[g2i[g], j] = 1.0
    A_gp_norm = (A_gp / np.sqrt(np.maximum(A_gp.sum(axis=0, keepdims=True), 1.0))).astype(np.float32)

    def make_mut_mask(target_gene_list: list) -> np.ndarray:
        M = np.zeros((len(driver_genes), len(target_gene_list)), dtype=np.float32)
        for i, d in enumerate(driver_genes):
            nbrs = drv_neighbors[d]
            for j, tg in enumerate(target_gene_list):
                if tg in nbrs:
                    M[i, j] = 1.0
        return M

    M_rppa = make_mut_mask(rppa_target_genes)
    M_prot = make_mut_mask(prot_genes)
    M_phos = make_mut_mask(phos_genes)
    print(f"  BioGRID TF Hubs={len(tf_list)} | KEGG Hubs={len(sel)} | Mut->RPPA density={M_rppa.mean()*100:.1f}%, Mut->Prot density={M_prot.mean()*100:.1f}%")
    return A_tf_norm, A_gp_norm, M_rppa, M_prot, M_phos


class LocalGated5OmicsPathwayGNN(nn.Module):
    """
    Optimized 5-Omics Pathway-GNN with Local Graph-Restricted Mutation & Low-Rank miRNA Gates:
      - Eliminates cross-cancer global mutation broadcasting noise!
      - Mutations ONLY modulate direct self & 1-hop BioGRID pathway neighbors via masked linear gates.
      - miRNA uses a tightly regularized 24-dim bottleneck with Modality Dropout.
    """
    def __init__(self, n_rna: int, n_mut: int, n_mir: int, rppa_cis: list, prot_cis: list, phos_cis: list,
                 A_tf: np.ndarray, A_gp: np.ndarray, M_rppa: np.ndarray, M_prot: np.ndarray, M_phos: np.ndarray, hid: int = 160):
        super().__init__()
        self.register_buffer("A_tf", torch.tensor(A_tf, dtype=torch.float32))
        self.register_buffer("A_gp", torch.tensor(A_gp, dtype=torch.float32))
        self.register_buffer("rppa_cis", torch.tensor(rppa_cis, dtype=torch.long))
        self.register_buffer("prot_cis", torch.tensor(prot_cis, dtype=torch.long))
        self.register_buffer("phos_cis", torch.tensor(phos_cis, dtype=torch.long))
        self.register_buffer("M_rppa", torch.tensor(M_rppa, dtype=torch.float32))
        self.register_buffer("M_prot", torch.tensor(M_prot, dtype=torch.float32))
        self.register_buffer("M_phos", torch.tensor(M_phos, dtype=torch.float32))

        self.rppa_cis_w = nn.Parameter(torch.ones(len(rppa_cis)) * 0.40)
        self.prot_cis_w = nn.Parameter(torch.ones(len(prot_cis)) * 0.55)
        self.phos_cis_w = nn.Parameter(torch.ones(len(phos_cis)) * 0.25)

        self.pw_enc = nn.Sequential(nn.Linear(A_gp.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.tf_enc = nn.Sequential(nn.Linear(A_tf.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.glob_enc = nn.Sequential(nn.Linear(n_rna, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.35))
        self.fuse_rna = nn.Sequential(nn.Linear(hid * 3, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.25))

        self.rppa_dec = nn.Linear(hid, len(rppa_cis))
        self.prot_dec = nn.Linear(hid, len(prot_cis))
        self.phos_dec = nn.Linear(hid, len(phos_cis))
        nn.init.zeros_(self.rppa_dec.weight)
        nn.init.zeros_(self.prot_dec.weight)
        nn.init.zeros_(self.phos_dec.weight)

        # Graph-restricted local mutation residual weights (zero outside 1-hop BioGRID neighborhood!)
        self.W_mut_rppa = nn.Parameter(torch.zeros(n_mut, len(rppa_cis)))
        self.W_mut_prot = nn.Parameter(torch.zeros(n_mut, len(prot_cis)))
        self.W_mut_phos = nn.Parameter(torch.zeros(n_mut, len(phos_cis)))

        # Compact regularized miRNA residual gate
        self.mir_enc = nn.Sequential(nn.Linear(n_mir, 24), nn.LayerNorm(24), nn.GELU(), nn.Dropout(0.35))
        self.mir_prot = nn.Linear(24, len(prot_cis), bias=False)
        self.mir_phos = nn.Linear(24, len(phos_cis), bias=False)
        nn.init.zeros_(self.mir_prot.weight)
        nn.init.zeros_(self.mir_phos.weight)

    def encode_rna(self, x_rna: torch.Tensor) -> torch.Tensor:
        return self.fuse_rna(torch.cat([self.pw_enc(x_rna @ self.A_gp), self.tf_enc(x_rna @ self.A_tf), self.glob_enc(x_rna)], dim=-1))

    def forward_rppa(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.rppa_cis >= 0).float()
        si = torch.clamp(self.rppa_cis, min=0)
        out = x_rna[:, si] * vm * self.rppa_cis_w + 0.35 * self.rppa_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_rppa * self.M_rppa)
        return out

    def forward_prot(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.prot_cis >= 0).float()
        si = torch.clamp(self.prot_cis, min=0)
        out = x_rna[:, si] * vm * self.prot_cis_w + 0.30 * self.prot_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_prot * self.M_prot)
        if x_mir is not None:
            out = out + 0.15 * self.mir_prot(self.mir_enc(x_mir))
        return out

    def forward_phos(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.phos_cis >= 0).float()
        si = torch.clamp(self.phos_cis, min=0)
        out = x_rna[:, si] * vm * self.phos_cis_w + 0.30 * self.phos_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_phos * self.M_phos)
        if x_mir is not None:
            out = out + 0.15 * self.mir_phos(self.mir_enc(x_mir))
        return out


def main():
    print(f"=== Exp27b: Local Graph-Gated 5-Omics Pathway-GNN Optimization ({DEVICE}) ===")

    hot = pd.read_csv(HOTSPOT_PATH)
    driver_genes = [str(c).split(" (")[0].strip().upper() for c in hot.columns[6:]]
    driver_set = set(driver_genes)
    drv_pos = {g: i for i, g in enumerate(driver_genes)}

    c24 = np.load(CACHE_24, allow_pickle=True)
    top_rna = [str(g) for g in c24["top_rna"]]
    top_prot = [str(g) for g in c24["top_prot"]]
    rna_pos = {g: i for i, g in enumerate(top_rna)}
    ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")

    # Load TCGA
    rppa = pd.read_csv(TCGA_RPPA_PATH, sep="\t", index_col=0)
    rppa = rppa[rppa.isna().mean(axis=1) < 0.30]
    rppa_abs = [str(a).strip() for a in rppa.index]
    rppa_target_genes = [RPPA_TO_GENE.get(ab, ab.split("_")[0].lstrip("X").upper()) for ab in rppa_abs]
    rppa_sids = set(rppa.columns)

    sid_to_mut = {}
    for mf in sorted(glob.glob(os.path.join(TCGA_MAF_DIR, "TCGA_*_mutect2_maf"))):
        df_m = pd.read_csv(mf, sep="\t", comment="#", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"], low_memory=False)
        df_m["Hugo_Symbol"] = df_m["Hugo_Symbol"].astype(str).str.strip().str.upper()
        df_m = df_m[(df_m["Variant_Classification"] != "Silent") & (df_m["Hugo_Symbol"].isin(driver_set))]
        for sym, bc in zip(df_m["Hugo_Symbol"].values, df_m["Tumor_Sample_Barcode"].astype(str).values):
            s15 = bc[:15]
            if s15 in rppa_sids:
                if s15 not in sid_to_mut:
                    sid_to_mut[s15] = np.zeros(len(driver_genes), dtype=np.float32)
                sid_to_mut[s15][drv_pos[sym]] = 1.0

    tx_l, tm_l, ty_l, tc_l = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(TCGA_RNA_DIR, "*_mrna_tpm.csv"))):
        ctype = os.path.basename(fp).replace("TCGA-", "").replace("_mrna_tpm.csv", "")
        df_r = pd.read_csv(fp, low_memory=False)
        df_r["gene"] = df_r["gene"].astype(str).str.strip().str.upper()
        df_r = df_r[df_r["gene"].isin(top_rna)].drop_duplicates("gene").set_index("gene")
        col_to_s15 = {c: str(c)[:15] for c in df_r.columns if str(c)[:15] in rppa_sids}
        if not col_to_s15:
            continue
        seen = {}
        for oc, s15 in col_to_s15.items():
            if s15 not in seen:
                seen[s15] = oc
        s15s = sorted(seen.keys())
        tx_l.append(row_zscore(np.log2(np.clip(df_r[[seen[s] for s in s15s]].reindex(top_rna).fillna(0.0).values.T, 0.0, None) + 1.0)))
        tm_l.append(np.stack([sid_to_mut.get(s, np.zeros(len(driver_genes), dtype=np.float32)) for s in s15s]))
        ty_l.append(np.nan_to_num(rppa[s15s].values.T.astype(np.float32), nan=0.0))
        tc_l.extend([ctype] * len(s15s))
    X_tcga_r, X_tcga_mut, Y_tcga_rppa, C_tcga = np.vstack(tx_l), np.vstack(tm_l), np.vstack(ty_l), np.array(tc_l)

    # Load CPTAC
    cptac_cancers = sorted(CPTAC_MUT_PREFIX.keys())
    sm, sph, cp_raw = None, None, {}
    for c in cptac_cancers:
        r = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_transcriptomics_bcm_processed_data.csv"), index_col=0)
        p = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_proteomics_bcm_processed_data.csv"), index_col=0)
        ph = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_phosphoproteomics_bcm_processed_data.csv"), index_col=0)
        r.index = [str(i).strip().upper() for i in r.index]; r = r.groupby(r.index, sort=False).mean()
        p.index = [str(i).strip().upper() for i in p.index]; p = p.groupby(p.index, sort=False).mean()
        ph.index = [str(i).split(":")[0].strip().upper() for i in ph.index]; ph = ph.groupby(ph.index, sort=False).mean()
        mir_p = os.path.join(CPTAC_DIR, f"{c}_miRNA_bcm_processed_data.csv")
        m_df = pd.read_csv(mir_p, index_col=0) if os.path.exists(mir_p) else None
        if m_df is not None:
            m_df.index = [str(i).strip().lower() for i in m_df.index]; m_df = m_df.groupby(m_df.index, sort=False).mean()
            sm = set(m_df.index) if sm is None else (sm & set(m_df.index))
        sph = set(ph.index) if sph is None else (sph & set(ph.index))
        mut_p = os.path.join(CPTAC_MUT_DIR, f"{CPTAC_MUT_PREFIX[c]}_somatic_mutation_gene_level_binary.txt")
        mut_df = pd.read_csv(mut_p, sep="\t", index_col=0)
        mut_df["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in mut_df.index]
        mut_df = mut_df[mut_df["__sym__"].isin(driver_set)].groupby("__sym__", sort=False).max()
        cp_raw[c] = (r, m_df, p, ph, mut_df)

    top_mir = sorted(list(cp_raw["brca"][1].loc[sorted(sm)].var(axis=1).sort_values(ascending=False).index[:350]))
    top_phos = sorted(list(cp_raw["brca"][3].loc[sorted(sph)].var(axis=1).sort_values(ascending=False).index[:800]))

    cx_l, cmut_l, cmir_l, chas_mir_l, cp_l, cph_l, cc_l = [], [], [], [], [], [], []
    for c in cptac_cancers:
        r, m_df, p, ph, mut_df = cp_raw[c]
        sids = sorted(set(r.columns) & set(p.columns) & set(ph.columns))
        cx_l.append(row_zscore(r[sids].reindex(top_rna).fillna(0.0).T.values))
        cp_l.append(row_zscore(p[sids].reindex(top_prot).fillna(0.0).T.values))
        cph_l.append(row_zscore(ph[sids].reindex(top_phos).fillna(0.0).T.values))
        cmut_l.append(mut_df.reindex(index=driver_genes, columns=sids).fillna(0.0).T.values.astype(np.float32))
        if m_df is not None:
            mir_mat = np.zeros((len(sids), len(top_mir)), dtype=np.float32)
            has_m = np.array([s in m_df.columns for s in sids], dtype=bool)
            vp = np.where(has_m)[0]
            if len(vp):
                mir_mat[vp] = row_zscore(m_df[[sids[i] for i in vp]].reindex(top_mir).fillna(0.0).T.values)
            cmir_l.append(mir_mat); chas_mir_l.append(has_m)
        else:
            cmir_l.append(np.zeros((len(sids), len(top_mir)), dtype=np.float32)); chas_mir_l.append(np.zeros(len(sids), dtype=bool))
        cc_l.extend([c.upper()] * len(sids))
    X_cp_r, X_cp_mut, X_cp_mir, has_cp_mir, Y_cp_prot, Y_cp_phos, C_cp = (
        np.vstack(cx_l), np.vstack(cmut_l), np.vstack(cmir_l), np.concatenate(chas_mir_l), np.vstack(cp_l), np.vstack(cph_l), np.array(cc_l)
    )

    # Load CCLE
    cell_rna_df = pd.read_csv(CELL_RNA_PATH, skiprows=[1, 2], low_memory=False)
    cell_rna_df = cell_rna_df.rename(columns={cell_rna_df.columns[0]: "gene_symbol"})
    cell_rna_df = cell_rna_df.drop(columns=[c for c in cell_rna_df.columns if str(c).startswith("Unnamed:")])
    cell_rna_df["gene_symbol"] = cell_rna_df["gene_symbol"].astype(str).str.strip().str.upper()
    cell_rna_df = cell_rna_df[cell_rna_df["gene_symbol"].isin(top_rna)].drop_duplicates("gene_symbol").set_index("gene_symbol").apply(pd.to_numeric, errors="coerce")
    cmp = pd.read_csv(CMP_PATH, usecols=["model_id", "model_name"]).dropna()
    name_to_mid = {norm_cell_name(mn): mid for mid, mn in zip(cmp["model_id"], cmp["model_name"]) if mid in cell_rna_df.columns}
    hot_clean = hot.drop_duplicates("ModelID").set_index("ModelID")
    hot_clean.columns = [str(c).split(" (")[0].strip().upper() for c in hot_clean.columns]
    hot_clean = hot_clean.reindex(columns=driver_genes).fillna(0.0)
    mid_to_mut = {mid: (hot_clean.loc[mid].values > 0).astype(np.float32) for mid in hot_clean.index}

    tmt_df = pd.read_csv(CCLE_TMT_PATH, low_memory=False)
    tmt_df["Gene_Symbol"] = tmt_df["Gene_Symbol"].astype(str).str.strip().str.upper()
    tmt_cols = [c for c in tmt_df.columns if "TenPx" in c and not c.endswith("_Peptides")]
    tmt_df = tmt_df[tmt_df["Gene_Symbol"].isin(top_prot)].groupby("Gene_Symbol", sort=False)[tmt_cols].mean()
    tmt_mids, tmt_cols_ok = [], []
    for col in tmt_cols:
        mid = name_to_mid.get(norm_cell_name(col.split("_")[0]))
        if mid is not None:
            tmt_mids.append(mid); tmt_cols_ok.append(col)
    X_ccle_tmt_r = row_zscore(cell_rna_df[tmt_mids].reindex(top_rna).fillna(0.0).T.values)
    X_ccle_tmt_mut = np.stack([mid_to_mut.get(m, np.zeros(len(driver_genes), dtype=np.float32)) for m in tmt_mids])
    Y_ccle_tmt = np.nan_to_num(row_zscore(tmt_df[tmt_cols_ok].reindex(top_prot).fillna(0.0).T.values), nan=0.0)

    ccle_rppa = pd.read_csv(CCLE_RPPA_PATH, index_col=0)
    tcga_norm_map = {norm_ab_name(a): j for j, a in enumerate(rppa_abs)}
    ccle_to_tcga_j = [(col, tcga_norm_map[norm_ab_name(col)]) for col in ccle_rppa.columns if norm_ab_name(col) in tcga_norm_map]
    cr_mids, cr_rows = [], []
    for idx_nm in ccle_rppa.index:
        mid = name_to_mid.get(norm_cell_name(str(idx_nm).split("_")[0]))
        if mid is not None:
            cr_mids.append(mid); cr_rows.append(idx_nm)
    X_ccle_rppa_r = row_zscore(cell_rna_df[cr_mids].reindex(top_rna).fillna(0.0).T.values)
    X_ccle_rppa_mut = np.stack([mid_to_mut.get(m, np.zeros(len(driver_genes), dtype=np.float32)) for m in cr_mids])
    Y_ccle_rppa = np.zeros((len(cr_rows), len(rppa_abs)), dtype=np.float32)
    M_ccle_rppa = np.zeros((len(cr_rows), len(rppa_abs)), dtype=np.float32)
    for cname, j_t in ccle_to_tcga_j:
        Y_ccle_rppa[:, j_t] = np.nan_to_num(ccle_rppa.loc[cr_rows, cname].values.astype(np.float32), nan=0.0)
        M_ccle_rppa[:, j_t] = 1.0

    # Graphs & Masks
    A_tf, A_gp, M_rppa, M_prot, M_phos = build_graphs_and_mut_masks(top_rna, driver_genes, rppa_target_genes, top_prot, top_phos)
    rppa_cis = [rna_pos.get(g, -1) for g in rppa_target_genes]
    prot_cis = [rna_pos.get(g, -1) for g in top_prot]
    phos_cis = [rna_pos.get(g, -1) for g in top_phos]

    tr_t = ~np.isin(C_tcga, list(UNSEEN_CANCERS))
    tr_c = ~np.isin(C_cp, list(UNSEEN_CANCERS))

    X_h1_r = np.vstack([X_tcga_r[tr_t], X_ccle_rppa_r])
    X_h1_mut = np.vstack([X_tcga_mut[tr_t], X_ccle_rppa_mut])
    Y_h1 = np.vstack([Y_tcga_rppa[tr_t], Y_ccle_rppa])
    M_h1 = np.vstack([np.ones_like(Y_tcga_rppa[tr_t]), M_ccle_rppa])

    X_h2_r = np.vstack([X_cp_r[tr_c], X_ccle_tmt_r])
    X_h2_mut = np.vstack([X_cp_mut[tr_c], X_ccle_tmt_mut])
    X_h2_mir = np.vstack([X_cp_mir[tr_c], np.zeros((len(X_ccle_tmt_r), len(top_mir)), dtype=np.float32)])
    H_h2_mir = np.concatenate([has_cp_mir[tr_c], np.zeros(len(X_ccle_tmt_r), dtype=bool)])
    Y_h2 = np.vstack([Y_cp_prot[tr_c], Y_ccle_tmt])

    X_h3_r, X_h3_mut, X_h3_mir, H_h3_mir, Y_h3 = X_cp_r[tr_c], X_cp_mut[tr_c], X_cp_mir[tr_c], has_cp_mir[tr_c], Y_cp_phos[tr_c]

    net = LocalGated5OmicsPathwayGNN(len(top_rna), len(driver_genes), len(top_mir), rppa_cis, prot_cis, phos_cis, A_tf, A_gp, M_rppa, M_prot, M_phos).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4)
    mse_fn = nn.MSELoss()

    l1 = DataLoader(TensorDataset(torch.tensor(X_h1_r), torch.tensor(X_h1_mut), torch.tensor(Y_h1), torch.tensor(M_h1)), batch_size=256, shuffle=True)
    l2 = DataLoader(TensorDataset(torch.tensor(X_h2_r), torch.tensor(X_h2_mut), torch.tensor(X_h2_mir), torch.tensor(H_h2_mir), torch.tensor(Y_h2)), batch_size=128, shuffle=True)
    l3 = DataLoader(TensorDataset(torch.tensor(X_h3_r), torch.tensor(X_h3_mut), torch.tensor(X_h3_mir), torch.tensor(H_h3_mir), torch.tensor(Y_h3)), batch_size=128, shuffle=True)

    net.train()
    for ep in range(35):
        for br, bm, by, bmask in l1:
            br, bm, by, bmask = br.to(DEVICE), bm.to(DEVICE), by.to(DEVICE), bmask.to(DEVICE)
            bm_in = None if np.random.rand() < 0.2 else bm
            opt.zero_grad()
            pred = net.forward_rppa(br, bm_in, None)
            loss = ((pred - by) ** 2 * bmask).sum() / torch.clamp(bmask.sum(), min=1.0) + 1e-3 * (net.W_mut_rppa ** 2).mean()
            loss.backward(); opt.step()
        for br, bm, bmi, bhasm, by in l2:
            br, bm, bmi, by = br.to(DEVICE), bm.to(DEVICE), bmi.to(DEVICE), by.to(DEVICE)
            bm_in = None if np.random.rand() < 0.2 else bm
            bmi_in = None if (not bhasm.any() or np.random.rand() < 0.25) else (bmi * bhasm.to(DEVICE).float().unsqueeze(-1))
            opt.zero_grad()
            loss = mse_fn(net.forward_prot(br, bm_in, bmi_in), by) + 1e-3 * (net.W_mut_prot ** 2).mean()
            loss.backward(); opt.step()
        for br, bm, bmi, bhasm, by in l3:
            br, bm, bmi, by = br.to(DEVICE), bm.to(DEVICE), bmi.to(DEVICE), by.to(DEVICE)
            bm_in = None if np.random.rand() < 0.2 else bm
            bmi_in = None if (not bhasm.any() or np.random.rand() < 0.25) else (bmi * bhasm.to(DEVICE).float().unsqueeze(-1))
            opt.zero_grad()
            loss = mse_fn(net.forward_phos(br, bm_in, bmi_in), by) + 1e-3 * (net.W_mut_phos ** 2).mean()
            loss.backward(); opt.step()

    net.eval()
    results = []
    with torch.no_grad():
        for c in sorted(UNSEEN_CANCERS):
            mt = (C_tcga == c)
            xt_r = torch.tensor(X_tcga_r[mt], device=DEVICE)
            xt_m = torch.tensor(X_tcga_mut[mt], device=DEVICE)
            for mode, xm_in in [("Full_LocalGated", xt_m), ("RNA_Only_Fallback", None)]:
                yp = net.forward_rppa(xt_r, xm_in, None).cpu().numpy()
                rs = [pearsonr(yp[:, j], Y_tcga_rppa[mt, j])[0] for j in range(len(rppa_abs)) if np.std(Y_tcga_rppa[mt, j]) > 1e-6]
                prs = [pearsonr(yp[:, j], Y_tcga_rppa[mt, j])[0] for j, ab in enumerate(rppa_abs) if ("_p" in ab and np.std(Y_tcga_rppa[mt, j]) > 1e-6)]
                results.append({"cancer": c, "platform": "TCGA_RPPA_All", "mode": mode, "median_r": float(np.median(rs))})
                results.append({"cancer": c, "platform": "TCGA_RPPA_Phospho", "mode": mode, "median_r": float(np.median(prs))})
                print(f"  [BLIND TCGA-{c:4s} | {mode:17s}] RPPA All Med r={np.median(rs):.4f} | RPPA Phospho Med r={np.median(prs):.4f}")

            mc = (C_cp == c)
            xc_r = torch.tensor(X_cp_r[mc], device=DEVICE)
            xc_m = torch.tensor(X_cp_mut[mc], device=DEVICE)
            xc_mi = torch.tensor(X_cp_mir[mc], device=DEVICE) if has_cp_mir[mc].any() else None
            for mode, xm_in, xmi_in in [("Full_LocalGated", xc_m, xc_mi), ("RNA_Only_Fallback", None, None)]:
                yp_p = net.forward_prot(xc_r, xm_in, xmi_in).cpu().numpy()
                yp_ph = net.forward_phos(xc_r, xm_in, xmi_in).cpu().numpy()
                rp = float(np.median([pearsonr(yp_p[:, j], Y_cp_prot[mc, j])[0] for j in range(len(top_prot)) if np.std(Y_cp_prot[mc, j]) > 1e-6]))
                rph = float(np.median([pearsonr(yp_ph[:, j], Y_cp_phos[mc, j])[0] for j in range(len(top_phos)) if np.std(Y_cp_phos[mc, j]) > 1e-6]))
                results.append({"cancer": c, "platform": "CPTAC_MS_Proteome", "mode": mode, "median_r": rp})
                results.append({"cancer": c, "platform": "CPTAC_MS_Phospho", "mode": mode, "median_r": rph})
                print(f"  [BLIND CPTAC-{c:4s} | {mode:17s}] MS Prot Med r={rp:.4f} | MS Phospho Med r={rph:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # Save improved checkpoint
    torch.save({
        "state_dict": net.state_dict(), "top_rna": top_rna, "top_prot": top_prot,
        "top_phos": top_phos, "top_mir": top_mir, "rppa_abs": rppa_abs, "driver_genes": driver_genes,
        "rppa_cis": rppa_cis, "prot_cis": prot_cis, "phos_cis": phos_cis,
        "A_tf": A_tf, "A_gp": A_gp, "M_rppa": M_rppa, "M_prot": M_prot, "M_phos": M_phos,
    }, OUT_CKPT)

    # Compare Exp27 (Global Broadcast) vs Exp27b (Local Graph-Gated)
    prev_df = pd.read_csv(r"D:\data\AIVC\test_results\exp27_unified_5omics_results.csv")
    prev_full = prev_df[prev_df["mode"] == "Full_MultiOmics"].copy()
    prev_full["model"] = "Exp27_GlobalBroadcast"
    curr_full = res_df[res_df["mode"] == "Full_LocalGated"].copy()
    curr_full["model"] = "Exp27b_LocalGraphGated"
    cmp_df = pd.concat([prev_full, curr_full], ignore_index=True)
    cmp_df["label"] = cmp_df["cancer"] + "\n" + cmp_df["platform"].str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=180)
    piv = cmp_df.pivot(index="label", columns="model", values="median_r")
    piv.plot(kind="bar", ax=ax, width=0.72, color=["#95a5a6", "#2980b9"], edgecolor="black", linewidth=0.5)
    ax.set_title("Exp27b Optimization: Global Broadcast vs Local Graph-Gated Multi-Omics\n(Blind Unseen Cancers LUAD & OV)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Median Cross-Patient Pearson (r)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp27b comparison to {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
