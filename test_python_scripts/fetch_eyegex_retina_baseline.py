import os
import gzip
import requests
import numpy as np
import pandas as pd

GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115828/suppl/GSE115828_Normalized_CPM.txt.gz"
OUT_DIR = r"D:\data\AIVC\data\eyegex"
GZ_PATH = os.path.join(OUT_DIR, "GSE115828_Normalized_CPM.txt.gz")
OUT_PARQUET = r"D:\data\AIVC\data\EyeGEx_GSE115828_retina_median.parquet"

os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(GZ_PATH):
    print(f"Downloading {GEO_URL} ...")
    with requests.get(GEO_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(GZ_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("Downloaded:", GZ_PATH, "size_mb=", round(os.path.getsize(GZ_PATH) / 1e6, 2))

df = pd.read_csv(GZ_PATH, sep="\t", compression="gzip", low_memory=False)
print("Shape:", df.shape)
print("First 6 columns:", list(df.columns[:6]))
print(df.iloc[:5, :5].to_string())

# 兼容第一列为 GeneID / Symbol，计算所有样本的中位表达作为 EyeGEx Retina 基线
id_col = df.columns[0]
num_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
med = num_df.median(axis=1)
res_df = pd.DataFrame({
    "gene_id": df[id_col].astype(str),
    "EyeGEx_Retina_Median_CPM": med.astype(np.float32),
})
res_df.to_parquet(OUT_PARQUET, index=False)
print(f"Saved EyeGEx Retina baseline -> {OUT_PARQUET} (genes={len(res_df)})")
