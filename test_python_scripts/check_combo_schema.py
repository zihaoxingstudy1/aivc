import numpy as np, pandas as pd
c_npz = np.load(r"D:\data\AIVC\data\combo_drug_kg_features.npz", allow_pickle=True)
print("combo_drug_kg_features keys:", list(c_npz.keys()))
for k in c_npz.keys():
    print(" ", k, c_npz[k].shape, c_npz[k][:3] if c_npz[k].ndim == 1 else "")

dc = pd.read_csv(r"D:\data\drug\summary_v_1_5.csv", nrows=3)
print("DrugComb cols:", list(dc.columns))
print(dc.iloc[:2].to_dict(orient="records"))
