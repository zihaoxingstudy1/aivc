import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex

# 1. Debug MCL1-KO NaN
adata_rep = ad.read_h5ad(r"D:\data\AIVC\data\ReplogleWeissman2022_K562_gwps.h5ad", backed="r")
mcl1_idx = np.where(adata_rep.obs["perturbation"].astype(str).values == "MCL1")[0]
nt_idx = np.where(adata_rep.obs["perturbation"].astype(str).values == "non-targeting")[0][:500]
print("Replogle MCL1 count:", len(mcl1_idx), "non-targeting count:", len(nt_idx))
if len(mcl1_idx) > 0:
    Xm = np.asarray(adata_rep.X[mcl1_idx].toarray() if sp.issparse(adata_rep.X[mcl1_idx]) else adata_rep.X[mcl1_idx], dtype=np.float32)
    Xn = np.asarray(adata_rep.X[nt_idx].toarray() if sp.issparse(adata_rep.X[nt_idx]) else adata_rep.X[nt_idx], dtype=np.float32)
    print("Xm nan count:", np.isnan(Xm).sum(), "Xn nan count:", np.isnan(Xn).sum())

# 2. Check Sciplex3 Panobinostat real deltas across doses vs GSE307106 (2h, 5h, 20h) & GSE308634
s3 = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad")
obs = s3.obs
X_s3 = s3.X.toarray().astype(np.float32) if sp.issparse(s3.X) else np.asarray(s3.X, dtype=np.float32)
ctrl_mask = obs["perturbation"].astype(str).str.lower().isin(["vehicle", "dmso", "control"])
x_ctrl = X_s3[ctrl_mask.values].mean(axis=0)
s3_genes = [str(g).upper() for g in s3.var_names]

ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")
df307 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE307106\GSE307106_raw_counts.txt.gz", sep="\t", index_col=0)
df307["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df307.index]
df307 = df307[df307["__sym__"] != ""].groupby("__sym__", sort=False).mean()
vals = df307.values
cpm = np.log2(vals / np.maximum(vals.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0)
mat307 = pd.DataFrame(cpm.T, index=df307.columns, columns=df307.index)

shared = [(j, g) for j, g in enumerate(s3_genes) if g in mat307.columns]
idx_s = [s[0] for s in shared]
cols_b = [s[1] for s in shared]

for t in ["2h", "5h", "20h"]:
    real_d = (mat307.loc[[f"Panobinostat_{t}_1", f"Panobinostat_{t}_2"], cols_b].mean(axis=0).values
              - mat307.loc[["DMSO_1", "DMSO_2"], cols_b].mean(axis=0).values)
    top50 = np.argsort(np.abs(real_d))[-50:]
    for dose in sorted(obs[obs["perturbation"].astype(str).str.contains("Panobinostat")]["dose_value"].unique()):
        pm = (obs["perturbation"].astype(str).str.contains("Panobinostat")) & (obs["dose_value"] == dose)
        sc_d = (X_s3[pm.values].mean(axis=0) - x_ctrl)[idx_s]
        r_all, _ = pearsonr(sc_d, real_d)
        r_t50, _ = pearsonr(sc_d[top50], real_d[top50])
        dir_t50 = np.mean(np.sign(sc_d[top50]) == np.sign(real_d[top50]))
        print(f"GSE307106 {t} vs Sciplex3 Panobinostat dose={dose}nM (n={pm.sum()}): All r={r_all:.4f} | Top50 r={r_t50:.4f}, DirAcc={dir_t50*100:.1f}%")
