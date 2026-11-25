import os, sys, pandas as pd, anndata as ad
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from geo_clinical_fetcher import download_file_if_needed, get_geo_series_prefix

RAW_BULK_DIR = r"D:\data\AIVC\data\wetlab_bulk_validation"

TARGETS = [
    ("GSE307106", "GSE307106_raw_counts.txt.gz", "\t"),
    ("GSE308634", "GSE308634_ALL_deseq_norm_counts_annotated.csv.gz", ","),
    ("GSE343784", "GSE343784_raw_counts.tsv.gz", "\t"),
]

for gse, fname, sep in TARGETS:
    prefix = get_geo_series_prefix(gse)
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/{fname}"
    dest = os.path.join(RAW_BULK_DIR, gse, fname)
    ok = download_file_if_needed(url, dest, timeout=180)
    print(f"Download {gse}/{fname}: {'OK' if ok else 'FAIL'}")
    if ok:
        df = pd.read_csv(dest, sep=sep, nrows=5)
        print(f"  cols[:8]={list(df.columns[:8])} (total={len(df.columns)})")
        print(df.iloc[:3, :5].to_string(index=False))

# Check drugs in sciplex3_drug_smiles.csv
sm = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
for kw in ["Panobinostat", "Doxorubicin", "Vorinostat", "Trametinib", "Paclitaxel"]:
    hits = sm[sm["perturbation"].str.contains(kw, case=False, na=False)]["perturbation"].tolist()
    print(f"sciplex3 match '{kw}': {hits}")

# Check MCL1 in ReplogleWeissman2022_K562_gwps.h5ad (backed='r')
adata_r = ad.read_h5ad(r"D:\data\AIVC\data\ReplogleWeissman2022_K562_gwps.h5ad", backed="r")
print("Replogle K562 obs cols:", list(adata_r.obs.columns))
for col in adata_r.obs.columns:
    vals = adata_r.obs[col].astype(str)
    if vals.str.contains("MCL1", case=False).any():
        print(f"  Replogle {col} MCL1 hits:", vals[vals.str.contains("MCL1", case=False)].unique()[:5])
