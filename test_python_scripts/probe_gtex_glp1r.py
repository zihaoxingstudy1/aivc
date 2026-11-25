import gzip
import pandas as pd

with gzip.open(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", "rt") as f:
    f.readline()
    f.readline()
    header = f.readline().strip().split("\t")
    organs = header[2:]
    print("GTEx total organs:", len(organs))
    eye_like = [c for c in organs if any(k in c.lower() for k in ["eye", "ret", "optic", "nerve", "artery", "pancreas"])]
    print("Eye/Nerve/Artery/Pancreas organs in GTEx:", eye_like)
    for line in f:
        parts = line.split("\t", 3)
        if parts[1].strip().upper() == "GLP1R":
            vals = [float(x) for x in line.strip().split("\t")[2:]]
            s = pd.Series(vals, index=organs).sort_values(ascending=False)
            print("\nTop 10 GTEx organs expressing GLP1R (TPM):")
            print(s.head(10).to_string())
            break

df = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
hits = df[df["perturbation"].astype(str).str.contains("sema|glp|liragl|tirzep", case=False, na=False)]
print("\nSemaglutide/GLP-1 hits in sciplex3_drug_smiles.csv:", len(hits))
