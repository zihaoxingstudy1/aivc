import pandas as pd
df = pd.read_csv(r"D:\data\tcga\protein\ccle-protein_quant_current_normalized.csv.gz", nrows=2)
cols = [c for c in df.columns if "TenPx" in c and not c.endswith("_Peptides")]
print("CCLE_TMT cell line cols count:", len(cols), "first 6:", cols[:6])
