import os
import sys

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from geo_clinical_fetcher import download_file_if_needed

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
GPL_DIR = os.path.join(RAW_DIR, "_GPL_annotations")
os.makedirs(GPL_DIR, exist_ok=True)

SUPPL_DOWNLOADS = [
    ("GSE194040", "GSE194nnn", "GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz"),
    ("GSE163882", "GSE163nnn", "GSE163882_all.data.tpms_222Samples.csv.gz"),
    ("GSE123845", "GSE123nnn", "GSE123845_exp_tpm_matrix.csv.gz"),
    ("GSE91061", "GSE91nnn", "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"),
    ("GSE176307", "GSE176nnn", "GSE176307_salmon_tpm_gene.matrix.tsv.gz"),
    ("GSE176307", "GSE176nnn", "GSE176307_BACI_Omniseq_Sample_Name_Key_submitted_GEO_v2.csv.gz"),
]

print("=== Downloading GEO Supplementary Expression Matrices ===")
for gse, prefix, fname in SUPPL_DOWNLOADS:
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/{fname}"
    dest = os.path.join(RAW_DIR, gse, fname)
    ok = download_file_if_needed(url, dest, timeout=300)
    size_mb = os.path.getsize(dest) / 1e6 if os.path.exists(dest) else 0
    print(f"  {gse}/{fname}: {'OK' if ok else 'FAIL'} ({size_mb:.2f} MB)")

# Also download GPL soft/annot tables for GPL96, GPL570, GPL97, GPL571
GPL_LIST = [
    ("GPL96", "GPLnnn"),
    ("GPL97", "GPLnnn"),
    ("GPL570", "GPLnnn"),
    ("GPL571", "GPLnnn"),
]
print("\n=== Downloading GEO Platform Annotations (GPL) ===")
for gpl, prefix in GPL_LIST:
    # GEO stores GPL annot.gz under /geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz
    fname = f"{gpl}.annot.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{gpl}/annot/{fname}"
    dest = os.path.join(GPL_DIR, fname)
    ok = download_file_if_needed(url, dest, timeout=180)
    size_mb = os.path.getsize(dest) / 1e6 if os.path.exists(dest) else 0
    print(f"  {fname}: {'OK' if ok else 'FAIL'} ({size_mb:.2f} MB)")
