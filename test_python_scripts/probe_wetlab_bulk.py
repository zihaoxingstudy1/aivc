import os, sys, re, ssl, urllib.request
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from geo_clinical_fetcher import download_geo_series_matrix, parse_geo_series_matrix_metadata_and_shape, get_geo_series_prefix

RAW_BULK_DIR = r"D:\data\AIVC\data\wetlab_bulk_validation"
ctx = ssl.create_default_context()

for gse in ["GSE307106", "GSE308634", "GSE343784", "GSE315547"]:
    print(f"\n=== Probing {gse} ===")
    files = download_geo_series_matrix(gse, RAW_BULK_DIR)
    for fp in files:
        meta, ng, probes, _ = parse_geo_series_matrix_metadata_and_shape(fp)
        meta.to_csv(fp.replace("_series_matrix.txt.gz", "_meta.csv"), index=False, encoding="utf-8-sig")
        print(f"  Matrix: {os.path.basename(fp)} | N={len(meta)} | Rows={ng} | Cols={list(meta.columns)}")
        print(f"  Titles[:4]: {list(meta['title'].head(4))}")
    prefix = get_geo_series_prefix(gse)
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/"
    try:
        html = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=20).read().decode("utf-8", "ignore")
        sfiles = sorted(set(f for f in re.findall(r'href="([^"]+)"', html) if not f.startswith("?") and not f.startswith("/") and "hhs.gov" not in f))
        print(f"  Suppl: {sfiles}")
    except Exception as e:
        print(f"  Suppl ERR: {e}")
