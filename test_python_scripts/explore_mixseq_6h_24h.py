import numpy as np
import scanpy as sc
from scipy.stats import pearsonr

ad = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad")
obs = ad.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
for c in obs.columns:
    obs[c] = obs[c].astype(str)

# 筛选 6h 与 24h 数据
sub = ad[obs["time"].isin(["6", "24"]).values].copy()
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)

so = sub.obs.copy()
for c in ["cell_line", "time", "perturbation", "dose_value"]:
    so[c] = so[c].astype(str)

ctrl_6 = {}
ctrl_24 = {}
for cl in so["cell_line"].unique():
    m6 = (so["cell_line"] == cl) & (so["time"] == "6") & (so["perturbation"] == "control")
    m24 = (so["cell_line"] == cl) & (so["time"] == "24") & (so["perturbation"] == "control")
    if m6.sum() >= 5:
        ctrl_6[cl] = np.asarray(sub.X[m6.values].mean(axis=0)).ravel()
    if m24.sum() >= 5:
        ctrl_24[cl] = np.asarray(sub.X[m24.values].mean(axis=0)).ravel()

tr = so[so["perturbation"] != "control"]
grp = tr.groupby(["cell_line", "perturbation", "dose_value", "time"]).indices
pairs = sorted({(k[0], k[1], k[2]) for k in grp.keys()})

rs, amps = [], []
for cl, pert, dose in pairs:
    if cl not in ctrl_6 or cl not in ctrl_24:
        continue
    i6 = grp.get((cl, pert, dose, "6"))
    i24 = grp.get((cl, pert, dose, "24"))
    if i6 is None or i24 is None or len(i6) < 15 or len(i24) < 15:
        continue
    d6 = np.asarray(sub.X[i6].mean(axis=0)).ravel() - ctrl_6[cl]
    d24 = np.asarray(sub.X[i24].mean(axis=0)).ravel() - ctrl_24[cl]
    n6, n24 = np.linalg.norm(d6), np.linalg.norm(d24)
    if n6 > 1e-6 and n24 > 1e-6:
        rs.append(float(pearsonr(d6, d24)[0]))
        amps.append(float(n24 / n6))

print(f"MIXseq 6h -> 24h 有效配对={len(rs)}, 药物={tr['perturbation'].nunique()}")
print(f"6h vs 24h 平均相关系数 r = {np.mean(rs):.4f} (median={np.median(rs):.4f}) | 幅值比 (24h/6h) = {np.mean(amps):.4f}")
