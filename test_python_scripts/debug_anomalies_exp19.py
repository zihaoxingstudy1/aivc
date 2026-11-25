import pandas as pd, sys
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import parse_gse9782_raw_chars

m123 = pd.read_csv(r"D:\data\AIVC\data\clinical_cohorts\raw\GSE123845\GSE123845_meta.csv")
print("=== GSE123845 pcr_status vs tumor_size & subtype & time ===")
print(m123.groupby("pcr_status")[["tumor_size", "regional_lymph_nodes"]].apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean()))
print(m123.groupby(["pcr_status", "subtype_ihc"]).size())

m9782 = parse_gse9782_raw_chars(r"D:\data\AIVC\data\clinical_cohorts\raw\GSE9782\GSE9782-GPL96_series_matrix.txt.gz")
print("\n=== GSE9782 did_patient_die & pgx_progression ===")
print("did_patient_die:", m9782["did_patient_die(0=no,1=yes)"].value_counts(dropna=False).to_dict())
print("pgx_progression:", m9782["pgx_progression(0=no,1=yes)"].value_counts(dropna=False).to_dict())

p123 = r"D:\data\AIVC\data\clinical_cohorts\cleaned\GSE123845\GSE123845_clinical.csv"
df_p123 = pd.read_csv(p123)
df_p123["response_binary"] = 1.0 - pd.to_numeric(df_p123["response_raw"], errors="coerce")
df_p123.to_csv(p123, index=False, encoding="utf-8-sig")
print("\nUpdated GSE123845 response_binary:", df_p123["response_binary"].value_counts().to_dict())
