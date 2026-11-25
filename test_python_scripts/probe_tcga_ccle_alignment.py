import os, glob, pandas as pd

# 1. Check TCGA RNA-seq TPM header & sample IDs
brca_rna = pd.read_csv(r"D:\data\tcga\rnaseq\TCGA-BRCA_mrna_tpm.csv", nrows=4)
print("TCGA-BRCA_mrna_tpm cols[:6]:", list(brca_rna.columns[:6]), "total cols:", len(brca_rna.columns))
print(brca_rna.iloc[:2, :4].to_string(index=False))

# Count all TCGA *_mrna_tpm.csv files
tpm_files = sorted(glob.glob(r"D:\data\tcga\rnaseq\*_mrna_tpm.csv"))
print("TCGA *_mrna_tpm.csv count:", len(tpm_files))

# 2. Check TCGA-RPPA-pancan-clean.xena sample IDs & antibody names
rppa = pd.read_csv(r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena", sep="\t", nrows=5)
print("TCGA-RPPA cols[:5]:", list(rppa.columns[:5]), "total:", len(rppa.columns))
rppa_idx = pd.read_csv(r"D:\data\tcga\protein\TCGA-RPPA-pancan-clean.xena", sep="\t", usecols=[0])
print("TCGA-RPPA antibody count:", len(rppa_idx), "sample[:15]:", list(rppa_idx.iloc[:15, 0]))

# 3. Check CCLE_RPPA_20180123 vs ccle-protein_quant sample IDs
ccle_rppa = pd.read_csv(r"D:\data\tcga\protein\CCLE_RPPA_20180123.csv", usecols=[0, 1, 2])
print("CCLE_RPPA rows:", len(ccle_rppa), "ids[:5]:", list(ccle_rppa.iloc[:5, 0]))
ccle_tmt = pd.read_csv(r"D:\data\tcga\protein\ccle-protein_quant_current_normalized.csv.gz", nrows=2)
print("CCLE_TMT cols[:12]:", list(ccle_tmt.columns[:12]))
