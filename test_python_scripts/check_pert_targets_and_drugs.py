import numpy as np, pandas as pd

t_npz = np.load(r"D:\data\AIVC\data\unified_esm2_targets.npz", allow_pickle=True)
print("unified_esm2_targets keys:", list(t_npz.keys()))
for k in t_npz.keys():
    print(" ", k, t_npz[k].shape, t_npz[k][:8] if t_npz[k].ndim == 1 else "")

m_npz = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=True)
print("sciplex3_mol_features keys:", list(m_npz.keys()))
for k in m_npz.keys():
    print(" ", k, m_npz[k].shape, m_npz[k][:10] if m_npz[k].ndim == 1 else "")

df_sm = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
print("sciplex3_drug_smiles cols:", list(df_sm.columns), "N=", len(df_sm))
print(df_sm.head(10).to_string(index=False))
