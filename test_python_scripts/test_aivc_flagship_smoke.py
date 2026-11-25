import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from aivc_flagship import AIVCFlagship

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AIVCFlagship(n_genes=2000, n_drug_feats=2048, n_mut=165).to(device)
model.eval()

x_e = torch.randn(4, 2000, device=device)
x_m = torch.randint(0, 2, (4, 165), device=device).float()
x_d1 = torch.randn(4, 2048, device=device)
x_d2 = torch.randn(4, 2048, device=device)

with torch.no_grad():
    ic50_with_mut = model.predict_ic50(x_e, x_d1, x_m)
    ic50_no_mut = model.predict_ic50(x_e, x_d1, None)
    syn = model.predict_combo_synergy(x_e, x_d1, x_d2, None)
    delta = model.predict_perturbation_delta(x_e, x_d1, None)

assert ic50_with_mut.shape == (4,) and ic50_no_mut.shape == (4,)
assert syn.shape == (4,) and delta.shape == (4, 2000)

# Test real clinical cohort loading through AIVCFlagship.load_omics
df_e = pd.read_parquet(r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE163882\GSE163882_expr_log2.parquet")
genes = list(df_e.columns[:2000])
sids, z_mat, mut_map = AIVCFlagship.load_omics(
    r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE163882\GSE163882_expr_log2.parquet",
    target_genes=genes,
)
print(f"AIVCFlagship smoke test PASSED on {device} | GSE163882 loaded: {z_mat.shape}, mut_map={len(mut_map)}")
