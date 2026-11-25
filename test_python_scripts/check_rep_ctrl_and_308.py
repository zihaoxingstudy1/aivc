import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import pearsonr
import sys
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex

adata_rep = ad.read_h5ad(r"D:\data\AIVC\data\ReplogleWeissman2022_K562_gwps.h5ad", backed="r")
print("Replogle top 10 perturbations:", adata_rep.obs["perturbation"].value_counts().head(10).to_dict())

# Also check GSE308634 vs Sciplex3 Panobinostat
s3 = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad")
obs = s3.obs
X_s3 = s3.X.toarray().astype(np.float32) if sp.issparse(s3.X) else np.asarray(s3.X, dtype=np.float32)
ctrl_mask = obs["perturbation"].astype(str).str.lower().isin(["vehicle", "dmso", "control"])
x_ctrl = X_s3[ctrl_mask.values].mean(axis=0)
s3_genes = [str(g).upper() for g in s3.var_names]

ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")
df308 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE308634\GSE308634_ALL_deseq_norm_counts_annotated.csv.gz", index_col=0)
num_cols = [c for c in df308.columns if ("_Veh_" in c or "_10nM_" in c)]
df308_num = df308[num_cols].copy()
df308_num["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df308_num.index]
df308_num = df308_num[df308_num["__sym__"] != ""].groupby("__sym__", sort=False).mean()
vals = df308_num.values
mat308 = pd.DataFrame(np.log2(vals / np.maximum(vals.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0).T, index=df308_num.columns, columns=df308_num.index)
veh_cols = [c for c in num_cols if "_Veh_" in c]
trt_cols = [c for c in num_cols if "_10nM_" in c]

shared = [(j, g) for j, g in enumerate(s3_genes) if g in mat308.columns]
idx_s = [s[0] for s in shared]
cols_b = [s[1] for s in shared]
real_d = mat308.loc[trt_cols, cols_b].mean(axis=0).values - mat308.loc[veh_cols, cols_b].mean(axis=0).values
top50 = np.argsort(np.abs(real_d))[-50:]
for dose in [10.0, 100.0, 1000.0, 10000.0]:
    pm = (obs["perturbation"].astype(str).str.contains("Panobinostat")) & (obs["dose_value"] == dose)
    sc_d = (X_s3[pm.values].mean(axis=0) - x_ctrl)[idx_s]
    r_all, _ = pearsonr(sc_d, real_d)
    r_t50, _ = pearsonr(sc_d[top50], real_d[top50])
    dir_t50 = np.mean(np.sign(sc_d[top50]) == np.sign(real_d[top50]))
    print(f"GSE308634 vs Sciplex3 Panobinostat {dose}nM: All r={r_all:.4f} | Top50 r={r_t50:.4f}, DirAcc={dir_t50*100:.1f}%")
