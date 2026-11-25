import numpy as np, pandas as pd
npz = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=True)
print("gdsc_drug_kg_features keys:", list(npz.keys()))
for k in npz.keys():
    print(" ", k, npz[k].shape, npz[k][:3] if npz[k].ndim == 1 else "")
df_g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", nrows=3)
print("GDSC2 cols:", list(df_g2.columns))
df_cmp = pd.read_csv(r"D:\data\AIVC\data\cmp_model_list.csv", nrows=3)
print("CMP cols:", list(df_cmp.columns[:8]))
df_rna = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", nrows=3)
print("RNAseq cols[:6]:", list(df_rna.columns[:6]))
