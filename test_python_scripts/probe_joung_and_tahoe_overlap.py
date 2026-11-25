# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import anndata as ad

print("=== 1. Probe JoungZhang2023_atlas.h5ad ===")
fp_j = r"D:\data\AIVC\data\JoungZhang2023_atlas.h5ad"
a = ad.read_h5ad(fp_j, backed="r")
print("shape:", a.shape)
print("obs columns:", list(a.obs.columns))
print("var columns:", list(a.var.columns))
print("var_names[:10]:", list(a.var_names[:10]))
vc = a.obs["perturbation"].astype(str).value_counts()
print("num unique perturbations:", len(vc))
print("top 15 perturbations:", vc.head(15).to_dict())
print("perturbation_type:", a.obs["perturbation_type"].astype(str).value_counts().to_dict())
print("uns keys:", list(a.uns.keys()) if hasattr(a, "uns") else [])
a.file.close()

print("\n=== 2. Probe NormanWeissman2019_filtered.h5ad (CRISPRa Overexpression) ===")
fp_n = r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad"
an = ad.read_h5ad(fp_n, backed="r")
print("shape:", an.shape, "perturbation_type:", an.obs["perturbation_type"].astype(str).value_counts().to_dict() if "perturbation_type" in an.obs else "N/A")
print("top 10 perts:", an.obs["perturbation"].astype(str).value_counts().head(10).to_dict())
an.file.close()

print("\n=== 3. Probe Tahoe-100M drug overlap with MIXseq ===")
mix = ad.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
mix_drugs = sorted(set(mix.obs["perturbation"].astype(str).unique()) - {"control"})
mix.file.close()
tahoe_drugs_df = pd.read_parquet(r"D:\data\AIVC\data\tahoe_100m\metadata\drug_metadata.parquet")
t_drugs = {str(d).strip().lower(): str(d).strip() for d in tahoe_drugs_df["drug"].dropna()}
overlap = [d for d in mix_drugs if d.lower() in t_drugs]
print(f"MIXseq drugs ({len(mix_drugs)}): {mix_drugs}")
print(f"Tahoe total drugs: {len(t_drugs)} | Exact overlap with MIXseq: {overlap}")
