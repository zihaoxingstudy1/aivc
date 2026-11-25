# -*- coding: utf-8 -*-
"""
小分子药物多模态结构特征提取模块 (ChemBERTa-77M-MTR + RDKit Morgan 指纹)
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@torch.no_grad()
def extract_drug_molecular_features(
    smiles_csv_path: str,
    chemberta_dir: str,
    save_npz_path: str,
    morgan_radius: int = 2,
    morgan_bits: int = 1024,
):
    """
    为 SMILES CSV 中的所有药物提取 ChemBERTa 语义嵌入与 RDKit Morgan 分子指纹。
    """
    df = pd.read_csv(smiles_csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(chemberta_dir)
    model = AutoModel.from_pretrained(chemberta_dir).to(device)
    model.eval()

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=int(morgan_radius), fpSize=int(morgan_bits))

    drug_names = df["perturbation"].astype(str).tolist()
    smiles_list = df["smiles"].fillna("").astype(str).tolist()

    # 1. RDKit Morgan 指纹
    morgan_feats = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            fp = mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
            valid_smiles.append(smi)
        else:
            fp = np.zeros(int(morgan_bits), dtype=np.float32)
            valid_smiles.append("C")
        morgan_feats.append(fp)
    morgan_mat = np.vstack(morgan_feats)

    # 2. ChemBERTa-77M-MTR 嵌入
    cb_feats = []
    for i in range(0, len(valid_smiles), 32):
        batch_smi = valid_smiles[i : i + 32]
        enc = tokenizer(batch_smi, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = model(**enc)
        # Mean pooling
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        cb_feats.append(emb.cpu().numpy().astype(np.float32))
    chemberta_mat = np.vstack(cb_feats)

    os.makedirs(os.path.dirname(os.path.abspath(save_npz_path)), exist_ok=True)
    np.savez(
        save_npz_path,
        drugs=np.array(drug_names),
        morgan=morgan_mat,
        chemberta=chemberta_mat,
    )
    print(f"[MolFeat] 已提取 {len(drug_names)} 种药物特征: Morgan={morgan_mat.shape}, ChemBERTa={chemberta_mat.shape} -> {save_npz_path}")
    return drug_names, morgan_mat, chemberta_mat


if __name__ == "__main__":
    extract_drug_molecular_features(
        smiles_csv_path=r"D:\data\AIVC\data\sciplex3_drug_smiles.csv",
        chemberta_dir=r"D:\data\AIVC\pretrained\chemberta",
        save_npz_path=r"D:\data\AIVC\data\sciplex3_mol_features.npz",
    )
