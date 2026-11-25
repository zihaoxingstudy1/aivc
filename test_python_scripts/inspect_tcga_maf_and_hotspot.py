import pandas as pd

# 1. Inspect DepMap 165 Hotspot driver gene list
hot = pd.read_csv(r"D:\data\drug\DeepMapcrisp\OmicsSomaticMutationsMatrixHotspot.csv", nrows=2)
print("Hotspot cols count:", len(hot.columns), "first 8:", list(hot.columns[:8]))

# 2. Inspect TCGA_BRCA_mutect2_maf header
with open(r"D:\TCGA\TCGA\mutation_data\TCGA_BRCA_mutect2_maf", "r", encoding="utf-8", errors="replace") as f:
    for i in range(10):
        line = f.readline().rstrip("\r\n")
        if not line.startswith("#"):
            print("MAF header:", line.split("\t")[:16])
            row1 = f.readline().rstrip("\r\n").split("\t")
            print("MAF row1:", row1[:16])
            break
