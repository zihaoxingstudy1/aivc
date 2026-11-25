import os
import pandas as pd

CLEAN_DIR = r"D:\data\AIVC\data\clinical_cohorts\cleaned"
inv = pd.read_csv(os.path.join(CLEAN_DIR, "cohort_inventory_summary.csv"))

shared_genes = None
all_clin = []
for cid in inv["cohort_id"]:
    cpath = os.path.join(CLEAN_DIR, cid, f"{cid}_clinical.csv")
    epath = os.path.join(CLEAN_DIR, cid, f"{cid}_expr_log2.parquet")
    c = pd.read_csv(cpath)
    e = pd.read_parquet(epath)
    assert len(c) == len(e) and list(c["sample_id"]) == list(e.index), f"Mismatch in {cid}"
    gset = set(e.columns)
    shared_genes = gset if shared_genes is None else (shared_genes & gset)
    all_clin.append(c)

big = pd.concat(all_clin, ignore_index=True)
print(f"Verified 17 cohorts, total aligned samples: {len(big)}")
print(f"Shared Gene Symbols across ALL 17 cohorts (Microarray + RNAseq): {len(shared_genes)}")
print("Platform breakdown:\n", big["platform_type"].value_counts().to_dict())
print("Timing breakdown:\n", big["treatment_timing"].value_counts().to_dict())
print("Multi-drug breakdown:\n", big["is_multi_drug"].value_counts().to_dict())
print("Disease breakdown:\n", big["disease"].value_counts().to_dict())
