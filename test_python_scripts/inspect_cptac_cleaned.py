import os, glob, pandas as pd

tmp_f = r"D:\data\CPTAC\cptac_processed_data\.._tmp_inspect.py"
if os.path.exists(tmp_f):
    os.remove(tmp_f)

CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
trans_files = sorted(glob.glob(os.path.join(CPTAC_DIR, "*_transcriptomics_bcm_processed_data.csv")))
print("Available CPTAC BCM cancer types:", [os.path.basename(f).split("_")[0] for f in trans_files])

for tag in ["transcriptomics_bcm", "proteomics_bcm", "phosphoproteomics_bcm", "CNV_bcm"]:
    p = os.path.join(CPTAC_DIR, f"brca_{tag}_processed_data.csv")
    df = pd.read_csv(p, nrows=3)
    print(f"brca_{tag}: cols[:6]={list(df.columns[:6])} (total={len(df.columns)})")
    print(df.iloc[:2, :4].to_string(index=False))

si = pd.read_csv(os.path.join(CPTAC_DIR, "brca_proteomics_bcm_sample_info.csv"), nrows=3)
print("sample_info cols:", list(si.columns), "\n", si.head(2).to_string(index=False))
