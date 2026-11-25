import os
import sys
import pandas as pd

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from geo_clinical_fetcher import (
    download_geo_series_matrix,
    parse_geo_series_matrix_metadata_and_shape,
)

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
OUT_SUMMARY = r"D:\data\AIVC\data\clinical_cohorts\cohort_probe_summary.csv"

CANDIDATE_GSES = [
    # Breast cancer NAC (multi-drug Taxane + Anthracycline + Cyclophosphamide +/- 5-FU)
    ("GSE25066", "Breast_Cancer", "Microarray"),
    ("GSE20194", "Breast_Cancer", "Microarray"),
    ("GSE32646", "Breast_Cancer", "Microarray"),
    ("GSE20271", "Breast_Cancer", "Microarray"),
    ("GSE22093", "Breast_Cancer", "Microarray"),
    ("GSE41998", "Breast_Cancer", "Microarray"),
    ("GSE194040", "Breast_Cancer_ISPY2", "Microarray"),
    ("GSE163882", "Breast_Cancer", "RNAseq"),
    ("GSE123845", "Breast_Cancer", "RNAseq"),
    # DLBCL / Lymphoma (R-CHOP / CHOP)
    ("GSE10846", "DLBCL", "Microarray"),
    ("GSE31312", "DLBCL", "Microarray"),
    ("GSE87371", "DLBCL", "Microarray"),
    # Multiple Myeloma (Bortezomib / Thalidomide / Dexamethasone)
    ("GSE9782", "Multiple_Myeloma", "Microarray"),
    ("GSE24080", "Multiple_Myeloma", "Microarray"),
    # Ovarian Cancer (Platinum + Taxane)
    ("GSE9891", "Ovarian_Cancer", "Microarray"),
    ("GSE26712", "Ovarian_Cancer", "Microarray"),
    # Colorectal Cancer (5-FU / FOLFOX / Cetuximab)
    ("GSE39582", "Colorectal_Cancer", "Microarray"),
    ("GSE5851", "Colorectal_Cancer", "Microarray"),
    # Immunotherapy RNA-seq
    ("GSE91061", "Melanoma_ICI", "RNAseq"),
    ("GSE176307", "Urothelial_ICI", "RNAseq"),
]

summary_rows = []
for gse_id, disease, expected_tech in CANDIDATE_GSES:
    print(f"\n=== Probing {gse_id} ({disease}, {expected_tech}) ===")
    files = download_geo_series_matrix(gse_id, RAW_DIR)
    if not files:
        print(f"  [MISS] No series_matrix downloaded for {gse_id}")
        continue
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            meta, n_genes, probes, vals = parse_geo_series_matrix_metadata_and_shape(fpath)
            meta_out = os.path.join(RAW_DIR, gse_id, fname.replace("_series_matrix.txt.gz", "_meta.csv"))
            meta.to_csv(meta_out, index=False, encoding="utf-8-sig")
            cols_str = "; ".join(list(meta.columns))
            print(f"  File: {fname} | Samples={len(meta)} | MatrixRows={n_genes} | FirstProbes={probes}")
            print(f"  Cols: {cols_str}")
            summary_rows.append({
                "gse_id": gse_id,
                "disease": disease,
                "expected_tech": expected_tech,
                "file": fname,
                "n_samples": len(meta),
                "n_matrix_rows": n_genes,
                "platforms": ";".join(sorted(meta["platform_id"].unique())),
                "meta_columns": cols_str,
            })
        except Exception as e:
            print(f"  [ERR] Failed parsing {fname}: {e}")

os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
print(f"\nSaved probe summary to {OUT_SUMMARY} ({len(summary_rows)} files)")
