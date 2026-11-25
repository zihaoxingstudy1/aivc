import os
import re
import ssl
import urllib.request
import pandas as pd

meta_paths = [
    r"D:\data\AIVC\data\clinical_cohorts\raw\GSE10846\GSE10846_meta.csv",
    r"D:\data\AIVC\data\clinical_cohorts\raw\GSE9782\GSE9782-GPL96_meta.csv",
    r"D:\data\AIVC\data\clinical_cohorts\raw\GSE24080\GSE24080_meta.csv",
    r"D:\data\AIVC\data\clinical_cohorts\raw\GSE26712\GSE26712_meta.csv",
    r"D:\data\AIVC\data\clinical_cohorts\raw\GSE123845\GSE123845_meta.csv",
]

for p in meta_paths:
    df = pd.read_csv(p)
    print(f"\n=== {os.path.basename(p)} (N={len(df)}) ===")
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique()[:4]
        print(f"  {col}: {list(vals)}")

ctx = ssl.create_default_context()
suppl_targets = [
    ("GSE194040", "GSE194nnn"),
    ("GSE163882", "GSE163nnn"),
    ("GSE123845", "GSE123nnn"),
    ("GSE91061", "GSE91nnn"),
    ("GSE176307", "GSE176nnn"),
]
print("\n=== GEO Supplementary Files ===")
for gse, prefix in suppl_targets:
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, context=ctx, timeout=20).read().decode("utf-8", "ignore")
        files = sorted(set(f for f in re.findall(r'href="([^"]+)"', html) if not f.startswith("?") and not f.startswith("/")))
        print(f"  {gse}: {files}")
    except Exception as e:
        print(f"  {gse} ERR: {e}")
