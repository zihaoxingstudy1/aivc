# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from chem_interface import compute_admet_druglikeness, map_molecules_to_targets_and_pathways

REF_CSV = r"D:\data\AIVC\data\sciplex3_drug_smiles.csv"
REF_NPZ = r"D:\data\AIVC\data\sciplex3_mol_features.npz"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
CHEMBERTA_DIR = r"D:\data\AIVC\pretrained\chemberta"
OUT_CSV = r"D:\data\AIVC\test_results\chem_interface_smoke_test.csv"

if __name__ == "__main__":
    # 选取 5 个真实肿瘤药物 (包含未在训练集重点使用的经典激酶/靶向/化疗药) 在 CPU 上验证接口
    test_mols = [
        ("Trametinib_MEK", "CC1=C2C(=C(N(C1=O)C)NC3=C(C=C(C=C3)I)F)C(=O)N(C(=O)N2C4=CC=CC(=C4)NC(=O)C)C5CC5"),
        ("Osimertinib_EGFR", "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C)OC"),
        ("Vemurafenib_BRAF", "CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)C4=CC=C(C=C4)Cl)F"),
        ("Palbociclib_CDK4_6", "CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C"),
        ("Panobinostat_HDAC", "CC1=C(C2=CC=CC=C2N1)CCNCC3=CC=C(C=C3)/C=C/C(=O)NO"),
    ]
    names = [m[0] for m in test_mols]
    smiles = [m[1] for m in test_mols]

    admet_df = compute_admet_druglikeness(smiles, names)
    map_df = map_molecules_to_targets_and_pathways(
        query_smiles=smiles,
        query_names=names,
        ref_smiles_csv=REF_CSV,
        ref_mol_npz=REF_NPZ,
        priors_dir=PRIORS_DIR,
        chemberta_dir=CHEMBERTA_DIR,
        top_k_ref=3,
        device_str="cpu",
    )
    merged = pd.merge(admet_df, map_df, on="drug_name")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("=== AI 制药化学接口 (ADMET + 靶点 + 全通路映射) CPU 实测结果 ===")
    cols = ["drug_name", "mw", "clogp", "qed", "lipinski_violations", "pains_alerts", "top_similar_ref_drugs", "inferred_targets", "mapped_prior_pathways"]
    print(merged[cols].to_string(index=False))
