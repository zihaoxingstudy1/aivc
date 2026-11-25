import os
import anndata as ad
import pandas as pd

DATA_DIR = r"D:\data\AIVC\data"

def check_time_pairs(fname, time_col="time", extra_cols=("cell_line", "perturbation", "dose_value")):
    fpath = os.path.join(DATA_DIR, fname)
    adata = ad.read_h5ad(fpath, backed="r")
    obs = adata.obs[[time_col, *extra_cols]].copy()
    adata.file.close()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)
    obs = obs[~obs[time_col].isin(["nan", "None", ""])]
    obs = obs[obs["perturbation"] != "control"]
    times = sorted(obs[time_col].unique())
    print(f"\n=== {fname} (非control有效细胞={len(obs)}) ===")
    print("时间点分布:", obs[time_col].value_counts().to_dict())

    grp = obs.groupby(list(extra_cols))[time_col].nunique()
    multi = grp[grp >= 2]
    print(f"拥有 >=2 个时间点的 (cell_line, perturbation, dose_value) 组合数: {len(multi)}")
    if len(multi) > 0:
        multi_idx = multi.index
        sub = obs.set_index(list(extra_cols)).loc[multi_idx].reset_index()
        ct = pd.crosstab(
            [sub["cell_line"], sub["perturbation"], sub["dose_value"]],
            sub[time_col]
        )
        print("跨时间点配对详情 (前10条):")
        print(ct.head(10).to_string())
        print(f"涉及细胞系数: {sub['cell_line'].nunique()}, 涉及药物数: {sub['perturbation'].nunique()}")

check_time_pairs("MIXseq_hvg2000.h5ad")
check_time_pairs("SrivatsanTrapnell2020_sciplex3.h5ad")
