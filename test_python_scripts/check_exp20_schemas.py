import anndata as ad, pandas as pd
s3 = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad", backed="r")
print("Sciplex3 shape:", s3.shape, "obs cols:", list(s3.obs.columns))
print("Sciplex3 drug col sample:", s3.obs.iloc[:3].to_dict(orient="records"))
print("Sciplex3 var[:5]:", list(s3.var_names[:5]))

df34 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE343784\GSE343784_raw_counts.tsv.gz", sep="\t", nrows=2)
print("GSE343784 cols:", list(df34.columns))
df307 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE307106\GSE307106_raw_counts.txt.gz", sep="\t", nrows=2)
print("GSE307106 cols:", list(df307.columns))
