import numpy as np, pandas as pd
npz = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=True)
gdsc_drugs = {str(d).upper(): str(d) for d in npz["drugs"]}
clin_drugs = [
    "Paclitaxel", "Docetaxel", "Doxorubicin", "Epirubicin", "Cyclophosphamide",
    "Fluorouracil", "5-Fluorouracil", "Carboplatin", "Cisplatin", "Oxaliplatin",
    "Irinotecan", "Bortezomib", "Ixabepilone", "Neratinib", "Veliparib",
    "Vincristine", "Bleomycin", "Dexamethasone", "MK-2206", "Trastuzumab",
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Rituximab", "Thalidomide"
]
matched = {d: gdsc_drugs[d.upper()] for d in clin_drugs if d.upper() in gdsc_drugs}
print("Matched in GDSC (542):", matched)
# Check rnaseq_merged_rsem_tpm gene column
df_r = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", usecols=[0, 1, 2, 3], nrows=5)
print("rnaseq first 5 rows:\n", df_r)
