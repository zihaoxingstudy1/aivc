import os, sys, pandas as pd
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_gpl_probe_and_entrez_maps, read_geo_series_matrix_expr, collapse_and_log2_samples_x_genes

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
GPL_DIR = os.path.join(RAW_DIR, "_GPL_annotations")
probe_maps, _ = load_gpl_probe_and_entrez_maps(GPL_DIR)
m = pd.read_csv(os.path.join(RAW_DIR, "GSE25066", "GSE25066_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE25066", "GSE25066_series_matrix.txt.gz"))
print("raw_e shape:", raw_e.shape)
print("raw_e index[:5]:", list(raw_e.index[:5]))
print("raw_e cols[:5]:", list(raw_e.columns[:5]))
print("m sample_id[:5]:", list(m["sample_id"].head(5)))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
print("e shape:", e.shape, "e index[:5]:", list(e.index[:5]))
common = [s for s in m["sample_id"] if s in e.index]
print("common count:", len(common))
print("pcr_rd values:", m["pathologic_response_pcr_rd"].value_counts(dropna=False).to_dict())
