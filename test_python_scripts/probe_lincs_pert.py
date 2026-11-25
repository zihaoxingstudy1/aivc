import pandas as pd
df = pd.read_csv(r"D:\data\AIVC\data\lincs_l1000\GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz", sep="\t", nrows=5, low_memory=False)
print("pert_info cols:", list(df.columns))
print(df.head(2).to_dict(orient="records"))
df_g = pd.read_csv(r"D:\data\AIVC\data\lincs_l1000\GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz", sep="\t", nrows=5, low_memory=False)
print("gene_info cols:", list(df_g.columns))
