import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import (
    load_gpl_probe_and_entrez_maps,
    load_ensembl_to_sym_from_gtex,
    read_geo_series_matrix_expr,
    collapse_and_log2_samples_x_genes,
    parse_gse9782_raw_chars,
    normalize_breast_regimen,
)

RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
CLEAN_DIR = r"D:\data\AIVC\data\clinical_cohorts\cleaned"
GPL_DIR = os.path.join(RAW_DIR, "_GPL_annotations")
GTEX_PATH = r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz"

os.makedirs(CLEAN_DIR, exist_ok=True)
probe_maps, entrez_to_sym = load_gpl_probe_and_entrez_maps(GPL_DIR)
ens_to_sym = load_ensembl_to_sym_from_gtex(GTEX_PATH)
print(f"Loaded GPL maps: { {k: len(v) for k, v in probe_maps.items()} } | Entrez: {len(entrez_to_sym)} | Ensembl: {len(ens_to_sym)}")


def save_clean_cohort(cid: str, clin_df: pd.DataFrame, expr_mat: pd.DataFrame, inventory: list):
    """Align clinical & expression 1-to-1, verify quality, and save into isolated cohort directory."""
    # Filter to valid sample_ids present in expr_mat
    clin_df = clin_df.dropna(subset=["sample_id"]).drop_duplicates(subset=["sample_id"]).copy()
    common_ids = [sid for sid in clin_df["sample_id"] if sid in expr_mat.index]
    clin_df = clin_df[clin_df["sample_id"].isin(common_ids)].copy()
    # Require at least one valid outcome (response_binary OR os_time_years/os_event OR pfs_time_years/pfs_event)
    has_outcome = (
        clin_df["response_binary"].notna()
        | (clin_df["os_time_years"].notna() & clin_df["os_event"].notna())
        | (clin_df["pfs_time_years"].notna() & clin_df["pfs_event"].notna())
        | clin_df["os_event"].notna()
        | clin_df["pfs_event"].notna()
    )
    clin_df = clin_df[has_outcome].reset_index(drop=True)
    expr_sub = expr_mat.loc[clin_df["sample_id"]].copy()

    # Check expression sanity
    assert list(expr_sub.index) == list(clin_df["sample_id"]), f"{cid} alignment mismatch!"
    out_dir = os.path.join(CLEAN_DIR, cid)
    os.makedirs(out_dir, exist_ok=True)
    clin_path = os.path.join(out_dir, f"{cid}_clinical.csv")
    expr_path = os.path.join(out_dir, f"{cid}_expr_log2.parquet")
    clin_df.to_csv(clin_path, index=False, encoding="utf-8-sig")
    expr_sub.to_parquet(expr_path)

    n_resp = int(clin_df["response_binary"].notna().sum())
    n_os = int((clin_df["os_time_years"].notna() & clin_df["os_event"].notna()).sum())
    n_pfs = int((clin_df["pfs_time_years"].notna() & clin_df["pfs_event"].notna()).sum())
    n_milestone = int((clin_df["os_event"].notna() | clin_df["pfs_event"].notna()).sum())
    inv = {
        "cohort_id": cid,
        "disease": clin_df["disease"].iloc[0],
        "platform_type": clin_df["platform_type"].iloc[0],
        "platform_id": ";".join(sorted(clin_df["platform_id"].astype(str).unique())),
        "n_samples": len(clin_df),
        "n_genes": expr_sub.shape[1],
        "n_with_response": n_resp,
        "n_with_os": n_os,
        "n_with_pfs": n_pfs,
        "n_with_any_survival_event": n_milestone,
        "timings": ";".join(sorted(clin_df["treatment_timing"].astype(str).unique())),
        "multi_drug_count": int((clin_df["is_multi_drug"] == 1).sum()),
        "expr_min": float(np.nanmin(expr_sub.values)),
        "expr_max": float(np.nanmax(expr_sub.values)),
    }
    inventory.append(inv)
    print(f"  [SAVED] {cid}: N={inv['n_samples']}, Genes={inv['n_genes']}, Resp={n_resp}, OS={n_os}, PFS={n_pfs}, Range=[{inv['expr_min']:.2f}, {inv['expr_max']:.2f}]")


inventory = []

# 1. GSE25066 (Breast Cancer, GPL96)
print("\nCleaning GSE25066...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE25066", "GSE25066_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE25066", "GSE25066_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
reg = m["type_taxane"].astype(str).apply(lambda x: f"{x} + Anthracycline + Cyclophosphamide")
drugs = reg.apply(normalize_breast_regimen)
c = pd.DataFrame({
    "cohort_id": "GSE25066", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": reg, "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pathologic_response_pcr_rd"],
    "response_binary": m["pathologic_response_pcr_rd"].map({"pCR": 1.0, "RD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan,
    "pfs_time_years": pd.to_numeric(m["drfs_even_time_years"], errors="coerce"),
    "pfs_event": pd.to_numeric(m["drfs_1_event_0_censored"], errors="coerce"),
    "endpoint_type": "pCR+DRFS",
})
save_clean_cohort("GSE25066", c, e, inventory)

# 2. GSE20194 (Breast Cancer, GPL96)
print("\nCleaning GSE20194...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE20194", "GSE20194_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE20194", "GSE20194_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
reg = m["treatment_code"].fillna("") + " | " + m["treatments_comments"].fillna("")
drugs = reg.apply(normalize_breast_regimen)
c = pd.DataFrame({
    "cohort_id": "GSE20194", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["treatment_code"].astype(str), "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pcr_vs_rd"],
    "response_binary": m["pcr_vs_rd"].map({"pCR": 1.0, "RD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE20194", c, e, inventory)

# 3. GSE32646 (Breast Cancer, GPL570)
print("\nCleaning GSE32646...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE32646", "GSE32646_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE32646", "GSE32646_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL570"])
c = pd.DataFrame({
    "cohort_id": "GSE32646", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": "Paclitaxel + FEC",
    "drugs_normalized": "Paclitaxel;Fluorouracil;Epirubicin;Cyclophosphamide",
    "is_multi_drug": 1,
    "response_raw": m["pathologic_response_pcr_ncr"],
    "response_binary": m["pathologic_response_pcr_ncr"].map({"pCR": 1.0, "nCR": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE32646", c, e, inventory)

# 4. GSE20271 (Breast Cancer, GPL96)
print("\nCleaning GSE20271...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE20271", "GSE20271_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE20271", "GSE20271_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
reg = m["preoperative_treatment"].fillna(m["treatment_received_(1=fac,_2=t/fac)"].astype(str))
drugs = reg.apply(normalize_breast_regimen)
c = pd.DataFrame({
    "cohort_id": "GSE20271", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": reg.astype(str), "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pcr_or_rd"],
    "response_binary": m["pcr_or_rd"].map({"pCR": 1.0, "RD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE20271", c, e, inventory)

# 5. GSE22093 (Breast Cancer, GPL96)
print("\nCleaning GSE22093...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE22093", "GSE22093_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE22093", "GSE22093_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
c = pd.DataFrame({
    "cohort_id": "GSE22093", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": "FAC/FEC",
    "drugs_normalized": "Fluorouracil;Doxorubicin;Cyclophosphamide",
    "is_multi_drug": 1,
    "response_raw": m["pcr.v.rd"],
    "response_binary": m["pcr.v.rd"].map({"pCR": 1.0, "RD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE22093", c, e, inventory)

# 6. GSE41998 (Breast Cancer, GPL571)
print("\nCleaning GSE41998...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE41998", "GSE41998_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE41998", "GSE41998_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL571"])
reg = m["treatment_arm"].astype(str).apply(lambda x: "Doxorubicin + Cyclophosphamide" if x == "none" else f"Doxorubicin + Cyclophosphamide + {x}")
drugs = reg.apply(normalize_breast_regimen)
resp_bin = m["pcr"].map({"Yes": 1.0, "No": 0.0})
c = pd.DataFrame({
    "cohort_id": "GSE41998", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": reg, "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pcr"],
    "response_binary": resp_bin,
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE41998", c, e, inventory)

# 7. GSE194040 (I-SPY2 Breast Cancer, 988 samples, Gene-level suppl)
print("\nCleaning GSE194040 (I-SPY2)...")
m1 = pd.read_csv(os.path.join(RAW_DIR, "GSE194040", "GSE194040-GPL20078_meta.csv"))
m2 = pd.read_csv(os.path.join(RAW_DIR, "GSE194040", "GSE194040-GPL30493_meta.csv"))
m = pd.concat([m1, m2], ignore_index=True)
m["patient_id_str"] = m["patient_id"].astype(str).str.strip()
raw_e = pd.read_csv(
    os.path.join(RAW_DIR, "GSE194040", "GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz"),
    sep="\t"
)
raw_e.columns = [str(col).strip() for col in raw_e.columns]
e = collapse_and_log2_samples_x_genes(raw_e, id_map=None)
# Map patient_id_str -> sample_id (GSM)
pid_to_gsm = dict(zip(m["patient_id_str"], m["sample_id"]))
e = e[e.index.isin(pid_to_gsm.keys())].copy()
e.index = [pid_to_gsm[idx] for idx in e.index]
drugs = m["arm"].astype(str).apply(normalize_breast_regimen)
c = pd.DataFrame({
    "cohort_id": "GSE194040", "sample_id": m["sample_id"], "patient_id": m["patient_id_str"],
    "disease": "Breast_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["arm"].astype(str), "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pcr"].astype(str),
    "response_binary": pd.to_numeric(m["pcr"], errors="coerce"),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE194040", c, e, inventory)

# 8. GSE163882 (Breast Cancer RNA-seq TPM, 222 samples)
print("\nCleaning GSE163882 (RNA-seq)...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE163882", "GSE163882_meta.csv"))
m["ba_id"] = m["title"].astype(str).apply(lambda x: x.split(":")[0].strip())
raw_e = pd.read_csv(os.path.join(RAW_DIR, "GSE163882", "GSE163882_all.data.tpms_222Samples.csv.gz"), index_col=0)
e = collapse_and_log2_samples_x_genes(raw_e, id_map=ens_to_sym)
ba_to_gsm = dict(zip(m["ba_id"], m["sample_id"]))
e = e[e.index.isin(ba_to_gsm.keys())].copy()
e.index = [ba_to_gsm[idx] for idx in e.index]
c = pd.DataFrame({
    "cohort_id": "GSE163882", "sample_id": m["sample_id"], "patient_id": m["ba_id"],
    "disease": "Breast_Cancer", "platform_type": "RNAseq", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": "Neoadjuvant_Chemotherapy", "drugs_normalized": "Taxane;Anthracycline;Cyclophosphamide",
    "is_multi_drug": 1,
    "response_raw": m["response_to_nac"],
    "response_binary": m["response_to_nac"].map({"pCR": 1.0, "RD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE163882", c, e, inventory)

# 9. GSE123845 (Breast Cancer RNA-seq TPM, paired T1/T2/T3, 227 samples)
print("\nCleaning GSE123845 (RNA-seq)...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE123845", "GSE123845_meta.csv"))
raw_e = pd.read_csv(os.path.join(RAW_DIR, "GSE123845", "GSE123845_exp_tpm_matrix.csv.gz"), index_col=0)
e = collapse_and_log2_samples_x_genes(raw_e, id_map=None)
title_to_gsm = dict(zip(m["title"].astype(str), m["sample_id"]))
e = e[e.index.isin(title_to_gsm.keys())].copy()
e.index = [title_to_gsm[idx] for idx in e.index]
timing_map = {"T1": "Pre-treatment", "T2": "On-treatment", "T3": "Post-treatment"}
drugs = m["nac_drug"].astype(str).apply(normalize_breast_regimen)
c = pd.DataFrame({
    "cohort_id": "GSE123845", "sample_id": m["sample_id"],
    "patient_id": m["title"].astype(str).apply(lambda x: "_".join(x.split("_")[:3])),
    "disease": "Breast_Cancer", "platform_type": "RNAseq", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": m["time"].map(timing_map).fillna("Unknown"),
    "regimen_raw": m["nac_drug"].astype(str), "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": m["pcr_status"].astype(str),
    # Note: In GSE123845, pcr_status=0 has tumor_size=0.0 (true pCR), pcr_status=1 has tumor_size=3.35cm (RD)
    "response_binary": pd.to_numeric(m["pcr_status"], errors="coerce").map({0: 1.0, 1: 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "pCR",
})
save_clean_cohort("GSE123845", c, e, inventory)

# 10. GSE10846 (DLBCL, GPL570, 420 samples)
print("\nCleaning GSE10846...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE10846", "GSE10846_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE10846", "GSE10846_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL570"])
reg_raw = m["clinical_info_9"].astype(str).str.replace("Chemotherapy:", "", regex=False).str.strip()
def norm_dlbcl(x):
    if "R-CHOP" in str(x):
        return "Rituximab;Cyclophosphamide;Doxorubicin;Vincristine;Prednisone"
    if "CHOP" in str(x):
        return "Cyclophosphamide;Doxorubicin;Vincristine;Prednisone"
    return "Unknown"
os_status = m["clinical_info_7"].astype(str).str.replace("Follow up status:", "", regex=False).str.strip()
os_years = pd.to_numeric(m["clinical_info_8"].astype(str).str.replace("Follow up years:", "", regex=False).str.strip(), errors="coerce")
drugs = reg_raw.apply(norm_dlbcl)
c = pd.DataFrame({
    "cohort_id": "GSE10846", "sample_id": m["sample_id"], "patient_id": m["individual"].astype(str),
    "disease": "DLBCL", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": reg_raw, "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": np.nan, "response_binary": np.nan,
    "os_time_years": os_years,
    "os_event": os_status.map({"DEAD": 1.0, "ALIVE": 0.0}),
    "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "OS",
})
save_clean_cohort("GSE10846", c, e, inventory)

# 11. GSE87371 (DLBCL, GPL570, 223 samples; cens_os=1 means censored, 0 means event!)
print("\nCleaning GSE87371...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE87371", "GSE87371_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE87371", "GSE87371_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL570"])
def norm_gse87371(x):
    s = str(x).upper()
    d = []
    if "R-" in s:
        d.append("Rituximab")
    if "CHOP" in s or "ACVBP" in s:
        d.extend(["Doxorubicin", "Cyclophosphamide", "Vincristine", "Prednisone"])
    if "ACVBP" in s:
        d.append("Bleomycin")
    return ";".join(dict.fromkeys(d))
drugs = m["treatment"].astype(str).apply(norm_gse87371)
c = pd.DataFrame({
    "cohort_id": "GSE87371", "sample_id": m["sample_id"], "patient_id": m["id"].astype(str),
    "disease": "DLBCL", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["treatment"].astype(str), "drugs_normalized": drugs,
    "is_multi_drug": 1,
    "response_raw": np.nan, "response_binary": np.nan,
    "os_time_years": pd.to_numeric(m["os_time"], errors="coerce") / 12.0,
    "os_event": 1.0 - pd.to_numeric(m["cens_os"], errors="coerce"),
    "pfs_time_years": pd.to_numeric(m["pfs_time"], errors="coerce") / 12.0,
    "pfs_event": 1.0 - pd.to_numeric(m["cens_pfs"], errors="coerce"),
    "endpoint_type": "OS+PFS",
})
save_clean_cohort("GSE87371", c, e, inventory)

# 12. GSE9782 (Multiple Myeloma, GPL96, 264 samples)
print("\nCleaning GSE9782...")
m = parse_gse9782_raw_chars(os.path.join(RAW_DIR, "GSE9782", "GSE9782-GPL96_series_matrix.txt.gz"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE9782", "GSE9782-GPL96_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
trt = m["treatment"].astype(str).map({"PS341": "Bortezomib", "Dex": "Dexamethasone"}).fillna(m["treatment"].astype(str))
c = pd.DataFrame({
    "cohort_id": "GSE9782", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Multiple_Myeloma", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["treatment"].astype(str), "drugs_normalized": trt,
    "is_multi_drug": 0,
    "response_raw": m["pgx_responder"].astype(str),
    "response_binary": m["pgx_responder"].map({"R": 1.0, "NR": 0.0}),
    "os_time_years": pd.to_numeric(m["days_survived_from_randomization"], errors="coerce") / 365.25,
    "os_event": pd.to_numeric(m["did_patient_die(0=no,1=yes)"], errors="coerce"),
    "pfs_time_years": pd.to_numeric(m["pgx_days_to_progression"], errors="coerce") / 365.25,
    "pfs_event": pd.to_numeric(m["pgx_progression(0=no,1=yes)"], errors="coerce"),
    "endpoint_type": "Response+OS+PFS",
})
save_clean_cohort("GSE9782", c, e, inventory)

# 13. GSE24080 (Multiple Myeloma, GPL570, 559 samples; exclude MAQC_Remove)
print("\nCleaning GSE24080...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE24080", "GSE24080_meta.csv"))
m = m[m["maqc_distribution_status"] != "MAQC_Remove"].copy()
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE24080", "GSE24080_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL570"])
os_24 = m["os_milestone_outcome_(24_months)"].astype(str).apply(lambda x: x.split(":")[-1].strip()).map({"1": 1.0, "0": 0.0})
efs_24 = m["efs_milestone_outcome_(24_months)"].astype(str).apply(lambda x: x.split(":")[-1].strip()).map({"1": 1.0, "0": 0.0})
c = pd.DataFrame({
    "cohort_id": "GSE24080", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Multiple_Myeloma", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": "Total_Therapy_2_or_3",
    "drugs_normalized": "Bortezomib;Thalidomide;Dexamethasone;Doxorubicin;Cyclophosphamide",
    "is_multi_drug": 1,
    "response_raw": np.nan, "response_binary": np.nan,
    "os_time_years": np.nan, "os_event": os_24,
    "pfs_time_years": np.nan, "pfs_event": efs_24,
    "endpoint_type": "OS_24m+EFS_24m",
})
save_clean_cohort("GSE24080", c, e, inventory)

# 14. GSE26712 (Ovarian Cancer, GPL96; exclude Normal HOSE)
print("\nCleaning GSE26712...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE26712", "GSE26712_meta.csv"))
m = m[~m["title"].astype(str).str.contains("Normal", case=False, na=False)].copy()
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE26712", "GSE26712_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL96"])
c = pd.DataFrame({
    "cohort_id": "GSE26712", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Ovarian_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": "Platinum_Taxane_Primary", "drugs_normalized": "Cisplatin;Paclitaxel",
    "is_multi_drug": 1,
    "response_raw": np.nan, "response_binary": np.nan,
    "os_time_years": pd.to_numeric(m["survival_years"], errors="coerce"),
    "os_event": m["status"].astype(str).apply(lambda x: 1.0 if "DOD" in x else (0.0 if ("AWD" in x or "NED" in x) else np.nan)),
    "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "OS",
})
save_clean_cohort("GSE26712", c, e, inventory)

# 15. GSE39582 (Colorectal Cancer, GPL570, 585 samples)
print("\nCleaning GSE39582...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE39582", "GSE39582_meta.csv"))
raw_e = read_geo_series_matrix_expr(os.path.join(RAW_DIR, "GSE39582", "GSE39582_series_matrix.txt.gz"))
e = collapse_and_log2_samples_x_genes(raw_e, probe_maps["GPL570"])
def norm_crc(row):
    if str(row["chemotherapy.adjuvant"]) != "Y":
        return "Untreated_Surgery_Only"
    t = str(row["chemotherapy.adjuvant.type"]).upper()
    if "FOLFOX" in t:
        return "Fluorouracil;Leucovorin;Oxaliplatin"
    if "FOLFIRI" in t:
        return "Fluorouracil;Leucovorin;Irinotecan"
    if "5FU" in t or "FUFOL" in t:
        return "Fluorouracil"
    return "Chemotherapy_Other"
drugs = m.apply(norm_crc, axis=1)
c = pd.DataFrame({
    "cohort_id": "GSE39582", "sample_id": m["sample_id"], "patient_id": m["title"].astype(str),
    "disease": "Colorectal_Cancer", "platform_type": "Microarray", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["chemotherapy.adjuvant"].astype(str) + ":" + m["chemotherapy.adjuvant.type"].fillna("None").astype(str),
    "drugs_normalized": drugs,
    "is_multi_drug": drugs.apply(lambda x: 1 if ";" in x else 0),
    "response_raw": np.nan, "response_binary": np.nan,
    "os_time_years": pd.to_numeric(m["os.delay_(months)"], errors="coerce") / 12.0,
    "os_event": pd.to_numeric(m["os.event"], errors="coerce"),
    "pfs_time_years": pd.to_numeric(m["rfs.delay"], errors="coerce") / 12.0,
    "pfs_event": pd.to_numeric(m["rfs.event"], errors="coerce"),
    "endpoint_type": "OS+RFS",
})
save_clean_cohort("GSE39582", c, e, inventory)

# 16. GSE91061 (Melanoma ICI RNA-seq FPKM, Pre vs On, 109 samples)
print("\nCleaning GSE91061 (RNA-seq)...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE91061", "GSE91061_meta.csv"))
raw_e = pd.read_csv(os.path.join(RAW_DIR, "GSE91061", "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"), index_col=0)
e = collapse_and_log2_samples_x_genes(raw_e, id_map=entrez_to_sym)
title_to_gsm = dict(zip(m["title"].astype(str), m["sample_id"]))
e = e[e.index.isin(title_to_gsm.keys())].copy()
e.index = [title_to_gsm[idx] for idx in e.index]
c = pd.DataFrame({
    "cohort_id": "GSE91061", "sample_id": m["sample_id"],
    "patient_id": m["title"].astype(str).apply(lambda x: x.split("_")[0]),
    "disease": "Melanoma_ICI", "platform_type": "RNAseq", "platform_id": m["platform_id"],
    "sample_type": "Tumor",
    "treatment_timing": m["visit_(pre_or_on_treatment)"].map({"Pre": "Pre-treatment", "On": "On-treatment"}).fillna("Unknown"),
    "regimen_raw": "Nivolumab", "drugs_normalized": "Nivolumab",
    "is_multi_drug": 0,
    "response_raw": m["response"].astype(str),
    "response_binary": m["response"].map({"PRCR": 1.0, "PD": 0.0, "SD": 0.0}),
    "os_time_years": np.nan, "os_event": np.nan, "pfs_time_years": np.nan, "pfs_event": np.nan,
    "endpoint_type": "RECIST_Response",
})
save_clean_cohort("GSE91061", c, e, inventory)

# 17. GSE176307 (Urothelial ICI RNA-seq TPM, 90 samples)
print("\nCleaning GSE176307 (RNA-seq)...")
m = pd.read_csv(os.path.join(RAW_DIR, "GSE176307", "GSE176307_meta.csv"))
m["baci_id"] = m["title"].astype(str).str.replace("Patient sample ", "", regex=False).str.strip()
key_df = pd.read_csv(os.path.join(RAW_DIR, "GSE176307", "GSE176307_BACI_Omniseq_Sample_Name_Key_submitted_GEO_v2.csv.gz"))
rs_to_baci = dict(zip(key_df["Omniseq_RS_ID (RNAseq)"].astype(str).str.strip(), key_df["Sample ID"].astype(str).str.strip()))
baci_to_gsm = dict(zip(m["baci_id"], m["sample_id"]))
raw_e = pd.read_csv(os.path.join(RAW_DIR, "GSE176307", "GSE176307_salmon_tpm_gene.matrix.tsv.gz"), sep="\t", index_col=0)
e = collapse_and_log2_samples_x_genes(raw_e, id_map=None)
mapped_idx = [baci_to_gsm.get(rs_to_baci.get(rs, ""), "") for rs in e.index]
e["__gsm__"] = mapped_idx
e = e[e["__gsm__"] != ""].set_index("__gsm__")
c = pd.DataFrame({
    "cohort_id": "GSE176307", "sample_id": m["sample_id"], "patient_id": m["baci_id"],
    "disease": "Urothelial_ICI", "platform_type": "RNAseq", "platform_id": m["platform_id"],
    "sample_type": "Tumor", "treatment_timing": "Pre-treatment",
    "regimen_raw": m["io.therapy"].astype(str), "drugs_normalized": m["io.therapy"].astype(str),
    "is_multi_drug": 0,
    "response_raw": m["io.response"].astype(str),
    "response_binary": m["io.response"].map({"CR": 1.0, "PR": 1.0, "SD": 0.0, "PD": 0.0}),
    "os_time_years": pd.to_numeric(m["overall_survival"], errors="coerce") / 365.25,
    "os_event": m["alive"].map({"No": 1.0, "Yes": 0.0}),
    "pfs_time_years": pd.to_numeric(m["pfs"], errors="coerce") / 365.25,
    "pfs_event": pd.to_numeric(m["progressed"], errors="coerce"),
    "endpoint_type": "RECIST+OS+PFS",
})
save_clean_cohort("GSE176307", c, e, inventory)

inv_df = pd.DataFrame(inventory)
inv_path = os.path.join(CLEAN_DIR, "cohort_inventory_summary.csv")
inv_df.to_csv(inv_path, index=False, encoding="utf-8-sig")
print("\n=== MASTER CLEANED CLINICAL COHORT INVENTORY ===")
print(inv_df[["cohort_id", "disease", "platform_type", "n_samples", "n_genes", "n_with_response", "n_with_os", "n_with_pfs", "multi_drug_count"]].to_string(index=False))
print(f"TOTAL CLEANED SAMPLES: {inv_df['n_samples'].sum()} across {len(inv_df)} isolated cohorts!")
