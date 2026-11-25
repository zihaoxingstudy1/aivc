import os, pandas as pd

files = [
    (r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena", "\t"),
    (r"D:\data\tcga\protein\CCLE_RPPA_20180123.csv", ","),
    (r"D:\data\tcga\protein\ccle-protein_quant_current_normalized.csv.gz", ","),
]
for fp, sep in files:
    df = pd.read_csv(fp, sep=sep, nrows=4)
    print(f"\n=== {os.path.basename(fp)} (size={os.path.getsize(fp)/1e6:.1f} MB) ===")
    print(f"  cols[:6]={list(df.columns[:6])} (total_cols={len(df.columns)})")
    print(df.iloc[:3, :4].to_string(index=False))
