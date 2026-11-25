import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

TCGA_RNA_DIR = r"D:\data\tcga\rnaseq"
TCGA_RPPA_PATH = r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena"
CCLE_RPPA_PATH = r"D:\data\tcga\protein\CCLE_RPPA_20180123.csv"
CCLE_TMT_PATH = r"D:\data\tcga\protein\ccle-protein_quant_current_normalized.csv.gz"
CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
CMP_PATH = r"D:\data\AIVC\data\cmp_model_list.csv"
CELL_RNA_PATH = r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv"
OUT_CACHE = r"D:\data\AIVC\data\exp24_multiomics_cache.npz"

CPTAC_CANCERS = ["brca", "ccrcc", "coad", "gbm", "hnscc", "lscc", "luad", "ov", "pdac", "ucec"]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def norm_cell_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def main():
    print("=== Preparing Exp24 Unified Multi-Omics Cache (TCGA + CPTAC + CCLE) ===")

    # 1. Determine shared RNA genes across TCGA, CPTAC, and Cell Line RNA-seq
    sample_tcga = pd.read_csv(r"D:\data\tcga\rnaseq\TCGA-BRCA_mrna_tpm.csv", usecols=[0])
    tcga_genes = set(sample_tcga.iloc[:, 0].astype(str).str.strip().str.upper())

    cptac_rna_genes = None
    cptac_prot_genes = None
    for c in CPTAC_CANCERS:
        rg = set(pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_transcriptomics_bcm_processed_data.csv"), usecols=[0]).iloc[:, 0].astype(str).str.strip().str.upper())
        pg = set(pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_proteomics_bcm_processed_data.csv"), usecols=[0]).iloc[:, 0].astype(str).str.strip().str.upper())
        cptac_rna_genes = rg if cptac_rna_genes is None else (cptac_rna_genes & rg)
        cptac_prot_genes = pg if cptac_prot_genes is None else (cptac_prot_genes & pg)

    cell_rna_df = pd.read_csv(CELL_RNA_PATH, skiprows=[1, 2], low_memory=False)
    cell_rna_df = cell_rna_df.rename(columns={cell_rna_df.columns[0]: "gene_symbol"})
    cell_rna_df = cell_rna_df.drop(columns=[c for c in cell_rna_df.columns if str(c).startswith("Unnamed:")])
    cell_rna_df["gene_symbol"] = cell_rna_df["gene_symbol"].astype(str).str.strip().str.upper()
    cell_rna_genes = set(cell_rna_df["gene_symbol"])

    shared_rna = sorted(tcga_genes & cptac_rna_genes & cell_rna_genes)
    # Select top 2,500 most variable RNA genes from cell lines + ensure all CPTAC top 1,200 proteins & RPPA genes are included
    cell_rna_sub = cell_rna_df[cell_rna_df["gene_symbol"].isin(shared_rna)].drop_duplicates("gene_symbol").set_index("gene_symbol").apply(pd.to_numeric, errors="coerce")
    top_rna = list(cell_rna_sub.var(axis=1).sort_values(ascending=False).index[:2200])

    # Load CCLE TMT to get shared deep-proteome genes
    tmt_df = pd.read_csv(CCLE_TMT_PATH, low_memory=False)
    tmt_df["Gene_Symbol"] = tmt_df["Gene_Symbol"].astype(str).str.strip().str.upper()
    tmt_cols = [c for c in tmt_df.columns if "TenPx" in c and not c.endswith("_Peptides")]
    tmt_df = tmt_df[tmt_df["Gene_Symbol"].isin(cptac_prot_genes)].groupby("Gene_Symbol", sort=False)[tmt_cols].mean()

    brca_p = pd.read_csv(os.path.join(CPTAC_DIR, "brca_proteomics_bcm_processed_data.csv"), index_col=0)
    brca_p.index = [str(i).strip().upper() for i in brca_p.index]
    brca_p = brca_p.groupby(brca_p.index, sort=False).mean()
    shared_prot = sorted(set(tmt_df.index) & set(brca_p.index))
    top_prot = sorted(list(brca_p.loc[shared_prot].var(axis=1).sort_values(ascending=False).index[:1200]))

    for g in top_prot:
        if g in shared_rna and g not in top_rna:
            top_rna.append(g)
    top_rna = sorted(top_rna)
    print(f"Unified Feature Space: RNA={len(top_rna)} | Deep Proteome={len(top_prot)}")

    # 2. Load TCGA RPPA & pair with all 33 TCGA *_mrna_tpm.csv files
    print("Loading TCGA-RPPA-pancan-clean.xena & 33 TCGA RNA-seq TPM cohorts...")
    rppa = pd.read_csv(TCGA_RPPA_PATH, sep="\t", index_col=0)
    rppa_ab_names = [str(a).strip() for a in rppa.index]
    # Keep antibodies with < 30% missingness
    valid_ab = rppa.isna().mean(axis=1) < 0.30
    rppa = rppa[valid_ab]
    rppa_ab_names = [str(a).strip() for a in rppa.index]
    rppa_sids = set(rppa.columns)

    tcga_x_list, tcga_y_list, tcga_cancer_list = [], [], []
    tpm_files = sorted(glob.glob(os.path.join(TCGA_RNA_DIR, "*_mrna_tpm.csv")))
    for fp in tpm_files:
        ctype = os.path.basename(fp).replace("TCGA-", "").replace("_mrna_tpm.csv", "")
        df_r = pd.read_csv(fp, low_memory=False)
        df_r["gene"] = df_r["gene"].astype(str).str.strip().str.upper()
        df_r = df_r[df_r["gene"].isin(top_rna)].drop_duplicates("gene").set_index("gene")
        # Map 28-char TCGA barcode -> 15-char sample ID (TCGA-XX-XXXX-01)
        col_to_sid15 = {c: str(c)[:15] for c in df_r.columns if str(c)[:15] in rppa_sids}
        if not col_to_sid15:
            continue
        # Deduplicate sid15 if multiple aliquots
        seen_s15 = {}
        for orig_c, s15 in col_to_sid15.items():
            if s15 not in seen_s15:
                seen_s15[s15] = orig_c
        s15_list = sorted(seen_s15.keys())
        orig_cols = [seen_s15[s] for s in s15_list]

        rna_mat = df_r[orig_cols].reindex(top_rna).fillna(0.0).values.T
        rna_log2 = np.log2(np.clip(rna_mat, 0.0, None) + 1.0)
        rna_z = row_zscore(rna_log2)

        y_rppa = rppa[s15_list].values.T.astype(np.float32)  # n x n_ab
        # Impute rare NaNs in RPPA with antibody median
        col_med = np.nanmedian(y_rppa, axis=0)
        inds = np.where(np.isnan(y_rppa))
        y_rppa[inds] = np.take(col_med, inds[1])

        tcga_x_list.append(rna_z)
        tcga_y_list.append(y_rppa)
        tcga_cancer_list.extend([ctype] * len(s15_list))

    X_tcga = np.vstack(tcga_x_list)
    Y_tcga_rppa = np.vstack(tcga_y_list)
    C_tcga = np.array(tcga_cancer_list)
    print(f"  TCGA Paired RNA + RPPA: {X_tcga.shape[0]} patients across {len(set(C_tcga))} cancers | RPPA antibodies={Y_tcga_rppa.shape[1]}")

    # 3. Load CPTAC 10 cancers (RNA + MS Proteome)
    print("Loading CPTAC 10 cancers...")
    cp_x_list, cp_y_list, cp_c_list = [], [], []
    for c in CPTAC_CANCERS:
        r = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_transcriptomics_bcm_processed_data.csv"), index_col=0)
        p = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_proteomics_bcm_processed_data.csv"), index_col=0)
        r.index = [str(i).strip().upper() for i in r.index]
        r = r.groupby(r.index, sort=False).mean()
        p.index = [str(i).strip().upper() for i in p.index]
        p = p.groupby(p.index, sort=False).mean()
        sids = sorted(set(r.columns) & set(p.columns))
        xr = row_zscore(r[sids].reindex(top_rna).fillna(0.0).T.values)
        yp = row_zscore(p[sids].reindex(top_prot).fillna(0.0).T.values)
        cp_x_list.append(xr)
        cp_y_list.append(yp)
        cp_c_list.extend([c.upper()] * len(sids))

    X_cptac = np.vstack(cp_x_list)
    Y_cptac_prot = np.vstack(cp_y_list)
    C_cptac = np.array(cp_c_list)
    print(f"  CPTAC Paired RNA + MS Proteome: {X_cptac.shape[0]} patients across {len(set(C_cptac))} cancers")

    # 4. Load CCLE Cell Lines (RNA + RPPA + TMT Proteome)
    print("Loading CCLE Cell Lines (RNA + RPPA + Gygi TMT)...")
    cmp = pd.read_csv(CMP_PATH, usecols=["model_id", "model_name"]).dropna()
    name_to_mid = {}
    for mid, mn in zip(cmp["model_id"], cmp["model_name"]):
        if mid in cell_rna_sub.columns:
            name_to_mid[norm_cell_name(mn)] = mid

    # CCLE TMT
    tmt_mids, tmt_cols_matched = [], []
    for col in tmt_cols:
        cl_prefix = norm_cell_name(col.split("_")[0])
        if cl_prefix in name_to_mid:
            tmt_mids.append(name_to_mid[cl_prefix])
            tmt_cols_matched.append(col)
    X_ccle_tmt = row_zscore(cell_rna_sub[tmt_mids].reindex(top_rna).fillna(0.0).T.values)
    Y_ccle_tmt = row_zscore(tmt_df[tmt_cols_matched].reindex(top_prot).fillna(0.0).T.values)
    print(f"  CCLE Paired RNA + TMT Proteome: {X_ccle_tmt.shape[0]} cell lines")

    # CCLE RPPA
    ccle_rppa = pd.read_csv(CCLE_RPPA_PATH, index_col=0)
    rppa_mids, rppa_rows_matched = [], []
    for idx_name in ccle_rppa.index:
        cl_prefix = norm_cell_name(str(idx_name).split("_")[0])
        if cl_prefix in name_to_mid:
            rppa_mids.append(name_to_mid[cl_prefix])
            rppa_rows_matched.append(idx_name)
    X_ccle_rppa = row_zscore(cell_rna_sub[rppa_mids].reindex(top_rna).fillna(0.0).T.values)
    Y_ccle_rppa = ccle_rppa.loc[rppa_rows_matched].values.astype(np.float32)
    print(f"  CCLE Paired RNA + RPPA: {X_ccle_rppa.shape[0]} cell lines x {Y_ccle_rppa.shape[1]} antibodies")

    np.savez_compressed(
        OUT_CACHE,
        top_rna=np.array(top_rna),
        top_prot=np.array(top_prot),
        tcga_rppa_abs=np.array(rppa_ab_names),
        ccle_rppa_abs=np.array(list(ccle_rppa.columns)),
        X_tcga=X_tcga,
        Y_tcga_rppa=Y_tcga_rppa,
        C_tcga=C_tcga,
        X_cptac=X_cptac,
        Y_cptac_prot=Y_cptac_prot,
        C_cptac=C_cptac,
        X_ccle_tmt=X_ccle_tmt,
        Y_ccle_tmt=Y_ccle_tmt,
        X_ccle_rppa=X_ccle_rppa,
        Y_ccle_rppa=Y_ccle_rppa,
    )
    total_samples = len(X_tcga) + len(X_cptac) + len(X_ccle_tmt) + len(X_ccle_rppa)
    print(f"\nSaved unified multi-omics cache to {OUT_CACHE} | Total paired records: {total_samples}")


if __name__ == "__main__":
    main()
