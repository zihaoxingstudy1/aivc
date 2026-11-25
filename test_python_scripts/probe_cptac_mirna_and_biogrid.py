import os, glob, pandas as pd

CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
mir_files = sorted(glob.glob(os.path.join(CPTAC_DIR, "*_miRNA_bcm_processed_data.csv")))
print("CPTAC BCM miRNA cancers:", [os.path.basename(f).split("_")[0] for f in mir_files])
df_m = pd.read_csv(mir_files[0], nrows=4)
print("Sample miRNA cols[:6]:", list(df_m.columns[:6]), "total cols:", len(df_m.columns))
print(df_m.iloc[:3, :4].to_string(index=False))

bg = pd.read_csv(r"D:\data\regulation\BIOGRID-ALL-4.4.211.tab3.txt", sep="\t", nrows=3, low_memory=False)
print("BioGRID cols[:18]:", list(bg.columns[:18]))
print(bg[["Official Symbol Interactor A", "Official Symbol Interactor B", "Organism Name Interactor A", "Organism Name Interactor B"]].head(3))
