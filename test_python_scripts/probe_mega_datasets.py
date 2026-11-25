# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import anndata as ad

print("=== 1. Tahoe-100M ===")
tahoe_dir = r"D:\data\AIVC\data\tahoe_100m"
for sub in sorted(glob.glob(os.path.join(tahoe_dir, "metadata", "*"))):
    print("  tahoe file:", os.path.basename(sub), f"({os.path.getsize(sub)/1e6:.1f} MB)")
    if sub.endswith(".parquet"):
        df = pd.read_parquet(sub)
        print("    shape:", df.shape, "cols:", list(df.columns)[:12])
        if len(df) > 0:
            print("    head(2):", df.head(2).to_dict(orient="records"))

print("\n=== 2. SchiebingerLander2019 & JoungZhang2023 ===")
for fname in ["SchiebingerLander2019_GSE115943.h5ad", "JoungZhang2023_atlas.h5ad"]:
    fp = os.path.join(r"D:\data\AIVC\data", fname)
    if os.path.exists(fp):
        a = ad.read_h5ad(fp, backed="r")
        print(f"  {fname}: shape={a.shape}, obs_cols={list(a.obs.columns)}")
        for col in a.obs.columns[:6]:
            vals = a.obs[col].astype(str).unique()[:5]
            print(f"    {col}: {vals}")
        a.file.close()

print("\n=== 3. LINCS L1000 (GSE70138 & GSE92742) ===")
lincs_dir = r"D:\data\AIVC\data\lincs_l1000"
for gse in ["GSE70138", "GSE92742"]:
    g_files = glob.glob(os.path.join(lincs_dir, f"{gse}*gene_info*.txt.gz"))
    s_files = glob.glob(os.path.join(lincs_dir, f"{gse}*sig_info*.txt.gz"))
    if g_files:
        gdf = pd.read_csv(g_files[0], sep="\t", nrows=3, low_memory=False)
        print(f"  {gse} gene_info cols:", list(gdf.columns))
    if s_files:
        sdf = pd.read_csv(s_files[0], sep="\t", nrows=5, low_memory=False)
        print(f"  {gse} sig_info cols:", list(sdf.columns))
        print(f"    sample row0:", sdf.iloc[0].to_dict())
