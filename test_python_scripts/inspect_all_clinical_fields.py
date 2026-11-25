import os
import pandas as pd

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"

df_i = pd.read_csv(
    os.path.join(RAW_DIR, "GSE194040", "GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz"),
    sep="\t", nrows=5
)
print("GSE194040 index[:5]:", list(df_i.index[:5]))

gtex = pd.read_csv(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", sep="\t", skiprows=2, usecols=[0, 1])
print("GTEx Ensembl->Symbol rows:", len(gtex), "sample:", gtex.head(3).to_dict(orient="records"))

# Inspect key clinical columns for each of the 17 cohorts
COHORT_META_FILES = [
    ("GSE25066", "GSE25066_meta.csv"),
    ("GSE20194", "GSE20194_meta.csv"),
    ("GSE32646", "GSE32646_meta.csv"),
    ("GSE20271", "GSE20271_meta.csv"),
    ("GSE22093", "GSE22093_meta.csv"),
    ("GSE41998", "GSE41998_meta.csv"),
    ("GSE194040", "GSE194040-GPL20078_meta.csv"),
    ("GSE194040_b", "GSE194040-GPL30493_meta.csv"),
    ("GSE163882", "GSE163882_meta.csv"),
    ("GSE123845", "GSE123845_meta.csv"),
    ("GSE10846", "GSE10846_meta.csv"),
    ("GSE87371", "GSE87371_meta.csv"),
    ("GSE9782", "GSE9782-GPL96_meta.csv"),
    ("GSE24080", "GSE24080_meta.csv"),
    ("GSE26712", "GSE26712_meta.csv"),
    ("GSE39582", "GSE39582_meta.csv"),
    ("GSE91061", "GSE91061_meta.csv"),
    ("GSE176307", "GSE176307_meta.csv"),
]

KEYWORDS = ("resp", "pcr", "os", "pfs", "drfs", "rfs", "efs", "surv", "dead", "alive",
            "status", "treat", "arm", "drug", "chemo", "taxane", "time", "visit", "io.", "title", "patient")

for cid, fname in COHORT_META_FILES:
    gse = cid.replace("_b", "")
    p = os.path.join(RAW_DIR, gse, fname)
    df = pd.read_csv(p)
    print(f"\n--- {cid} ({fname}, N={len(df)}) ---")
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in KEYWORDS) or cl.startswith("char_") or cl.startswith("clinical_info"):
            uniq = df[col].dropna().astype(str).unique()
            print(f"  {col} (n_uniq={len(uniq)}): {list(uniq[:5])}")
