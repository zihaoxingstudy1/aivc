import os
import anndata as ad

DATA_DIR = r"D:\data\AIVC\data"
FILES = [
    "AissaBenevolenskaya2021.h5ad",
    "SrivatsanTrapnell2020_sciplex2.h5ad",
    "DixitRegev2016_K562_TFs_7_days.h5ad",
    "DixitRegev2016_K562_TFs_13_days.h5ad",
]

for fn in FILES:
    fp = os.path.join(DATA_DIR, fn)
    if not os.path.exists(fp):
        print(f"[SKIP] {fn}")
        continue
    a = ad.read_h5ad(fp, backed="r")
    print(f"\n[{fn}] shape={a.shape}")
    print("  obs cols:", list(a.obs.columns))
    for col in a.obs.columns:
        low = col.lower()
        if any(k in low for k in ["time", "day", "hour", "pert", "drug", "dose", "cond", "cell"]):
            vc = a.obs[col].astype(str).value_counts()
            print(f"    {col} (n={vc.shape[0]}): {dict(list(vc.items())[:6])}")
    a.file.close()
