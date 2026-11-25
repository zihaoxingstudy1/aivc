import os
import gzip
import pandas as pd

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
GPL_DIR = os.path.join(RAW_DIR, "_GPL_annotations")

print("=== 1. Inspecting GPL Annotations ===")
for gpl in ["GPL96", "GPL97", "GPL570", "GPL571"]:
    p = os.path.join(GPL_DIR, f"{gpl}.annot.gz")
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        rows = []
        for line in f:
            if line.startswith("!platform_table_begin"):
                header = next(f).rstrip("\r\n").split("\t")
                continue
            if header is not None:
                if line.startswith("!platform_table_end"):
                    break
                rows.append(line.rstrip("\r\n").split("\t"))
                if len(rows) >= 3:
                    break
    print(f"  {gpl} cols: {header[:6]}")
    print(f"    row0: {rows[0][:6] if rows else []}")

print("\n=== 2. Inspecting Supplementary Expression Files ===")
suppl_files = [
    ("GSE194040", "GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz", "\t"),
    ("GSE163882", "GSE163882_all.data.tpms_222Samples.csv.gz", ","),
    ("GSE123845", "GSE123845_exp_tpm_matrix.csv.gz", ","),
    ("GSE91061", "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz", ","),
    ("GSE176307", "GSE176307_salmon_tpm_gene.matrix.tsv.gz", "\t"),
    ("GSE176307", "GSE176307_BACI_Omniseq_Sample_Name_Key_submitted_GEO_v2.csv.gz", ","),
]
for gse, fname, sep in suppl_files:
    p = os.path.join(RAW_DIR, gse, fname)
    df = pd.read_csv(p, sep=sep, nrows=5)
    print(f"  {gse}/{fname}: cols[:6]={list(df.columns[:6])} (total cols={len(df.columns)})")
    print(f"    first 3 rows col0..2:\n{df.iloc[:3, :3].to_string(index=False)}")
