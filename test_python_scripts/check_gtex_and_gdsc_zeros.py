import numpy as np, pandas as pd

gtex = pd.read_csv(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", sep="\t", skiprows=2, nrows=1)
print("GTEx organ cols (54):", list(gtex.columns[2:]))

npz = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=True)
d2f = {str(d).upper(): npz["feats"][i] for i, d in enumerate(npz["drugs"])}
for d in ["Bortezomib", "Doxorubicin", "Epirubicin", "Paclitaxel", "Cisplatin", "Docetaxel", "Methotrexate", "Gemcitabine", "Cyclophosphamide", "5-Fluorouracil", "Oxaliplatin", "Dabrafenib", "Erlotinib", "Osimertinib", "Lapatinib", "Gefitinib"]:
    f = d2f.get(d.upper(), None)
    print(f"  {d:18s}: sum={f.sum() if f is not None else 'MISS'} | l2={np.linalg.norm(f) if f is not None else 'MISS'}")

c_npz = np.load(r"D:\data\AIVC\data\combo_drug_kg_features.npz", allow_pickle=True)
cd2f = {str(d).upper(): c_npz["feats"][i] for i, d in enumerate(c_npz["drugs"])}
for d in ["Paclitaxel", "Doxorubicin", "Cisplatin", "Gemcitabine", "Cyclophosphamide", "5-Fluorouracil", "Oxaliplatin", "Dabrafenib", "Trametinib", "Vorinostat", "Bortezomib"]:
    f = cd2f.get(d.upper(), None)
    print(f"  COMBO {d:16s}: sum={f.sum() if f is not None else 'MISS'} | l2={np.linalg.norm(f) if f is not None else 'MISS'}")
