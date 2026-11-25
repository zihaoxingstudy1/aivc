import scanpy as sc
import pandas as pd

ad = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
obs = ad.obs[["perturbation", "dose_value", "time", "cell_line"]].copy()
ad.file.close()
for c in obs.columns:
    obs[c] = obs[c].astype(str)

sub = obs[obs["time"].isin(["6", "24"])].copy()
ct = pd.crosstab([sub["perturbation"], sub["dose_value"]], sub["time"])
ct["n_cell_lines_6h"] = sub[sub["time"] == "6"].groupby(["perturbation", "dose_value"])["cell_line"].nunique()
ct["n_cell_lines_24h"] = sub[sub["time"] == "24"].groupby(["perturbation", "dose_value"])["cell_line"].nunique()
ct = ct.fillna(0).astype(int)
print("=== MIXseq 6h 与 24h 具体处理药物、剂量、细胞数及覆盖细胞系数 ===")
print(ct.to_string())
