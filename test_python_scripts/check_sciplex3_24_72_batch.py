import numpy as np
import scanpy as sc
from scipy.stats import pearsonr

ad = sc.read_h5ad(r"D:\data\AIVC\data\SrivatsanTrapnell2020_sciplex3.h5ad", backed="r")
obs = ad.obs[["cell_line", "time", "plate", "perturbation", "dose_value"]].copy()
for c in obs.columns:
    obs[c] = obs[c].astype(str)
m = (obs["cell_line"] == "A549") & (obs["time"].isin(["24.0", "72.0"]))
sub_obs = obs[m]
print("A549 24h plates:", sub_obs[sub_obs["time"] == "24.0"]["plate"].value_counts().to_dict())
print("A549 72h plates:", sub_obs[sub_obs["time"] == "72.0"]["plate"].value_counts().to_dict())
print("A549 24h control plates:", sub_obs[(sub_obs["time"] == "24.0") & (sub_obs["perturbation"] == "control")]["plate"].value_counts().to_dict())
print("A549 72h control plates:", sub_obs[(sub_obs["time"] == "72.0") & (sub_obs["perturbation"] == "control")]["plate"].value_counts().to_dict())
ad.file.close()
