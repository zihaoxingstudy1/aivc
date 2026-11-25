import os
import glob
import anndata as ad
import pandas as pd

DATA_DIR = r"D:\data\AIVC\data"

h5ad_files = [
    "SrivatsanTrapnell2020_sciplex3.h5ad",
    "SrivatsanTrapnell2020_sciplex4.h5ad",
    "McFarlandTsherniak2020.h5ad",
    "MIXseq_hvg2000.h5ad",
    "NormanWeissman2019_filtered.h5ad",
]

print("=== 1. 探查单细胞扰动数据集的 obs 列与时间点/剂量/细胞系分布 ===")
for fname in h5ad_files:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] 不存在: {fpath}")
        continue
    adata = ad.read_h5ad(fpath, backed="r")
    print(f"\n[{fname}] shape={adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    for col in adata.obs.columns:
        low = col.lower()
        if any(k in low for k in ["time", "hour", "dose", "conc", "cell", "line", "treatment", "drug", "pert", "combo"]):
            vc = adata.obs[col].astype(str).value_counts()
            top_items = dict(list(vc.items())[:8])
            print(f"    - {col} (nunique={vc.shape[0]}): {top_items}")
    adata.file.close()

print("\n=== 2. 探查 clinical_cohorts 中的时间点/纵向样本 ===")
clin_files = glob.glob(os.path.join(DATA_DIR, "clinical_cohorts", "*.parquet"))
print(f"clinical parquet count: {len(clin_files)}")
for cf in sorted(clin_files)[:10]:
    df = pd.read_parquet(cf)
    cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "visit", "treat", "stage", "patient", "sample"])]
    if cols:
        print(f"  {os.path.basename(cf)} shape={df.shape} meta_cols={cols[:10]}")
