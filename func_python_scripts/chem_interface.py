# -*- coding: utf-8 -*-
"""
chem_interface.py - AI 制药化学接口模块 (ADMET 成药性评估 + 分子-靶点-全通路映射接口)
完全解耦于虚拟细胞主训练流程，负责将任意化学小分子 SMILES 转换为：
  1. RDKit 标准成药性与 ADMET 理化指标 (MW, cLogP, TPSA, HBD/HBA, QED, Lipinski Ro5, Veber, BBB, PAINS 警报)
  2. 基于 Morgan 指纹 + ChemBERTa 语义嵌入的跨库药物-靶点近邻推断 (DTI Proxy)
  3. 靶点自动映射至全量人类通路先验底座 (D:\\data\\AIVC\\data\\priors)
"""

import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, FilterCatalog, rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel


def _build_pains_catalog():
    """构建 RDKit 标准 PAINS 泛筛选干扰结构过滤目录。"""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


def compute_admet_druglikeness(smiles_list: list, names: list = None) -> pd.DataFrame:
    """
    批量计算小分子 SMILES 的成药性与 ADMET 核心理化指标。
    """
    if names is None:
        names = [f"Mol_{i}" for i in range(len(smiles_list))]
    pains_cat = _build_pains_catalog()
    rows = []

    for name, smi in zip(names, smiles_list):
        smi_str = "" if smi is None else str(smi).strip()
        mol = Chem.MolFromSmiles(smi_str) if smi_str else None
        if mol is None:
            rows.append({
                "drug_name": name,
                "smiles": smi_str,
                "valid_smiles": False,
                "mw": np.nan,
                "clogp": np.nan,
                "tpsa": np.nan,
                "hbd": np.nan,
                "hba": np.nan,
                "rot_bonds": np.nan,
                "qed": np.nan,
                "lipinski_violations": np.nan,
                "oral_veber_pass": False,
                "bbb_permeable_est": False,
                "pains_alerts": np.nan,
            })
            continue

        mw = float(Descriptors.MolWt(mol))
        clogp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rot = int(Lipinski.NumRotatableBonds(mol))
        qed_val = float(QED.qed(mol))

        ro5_viol = int(mw > 500.0) + int(clogp > 5.0) + int(hbd > 5) + int(hba > 10)
        veber_ok = bool(tpsa <= 140.0 and rot <= 10)
        bbb_ok = bool(mw <= 450.0 and tpsa <= 90.0 and 1.0 <= clogp <= 4.0)
        pains_hits = len(pains_cat.GetMatches(mol))

        rows.append({
            "drug_name": name,
            "smiles": smi_str,
            "valid_smiles": True,
            "mw": round(mw, 2),
            "clogp": round(clogp, 3),
            "tpsa": round(tpsa, 2),
            "hbd": hbd,
            "hba": hba,
            "rot_bonds": rot,
            "qed": round(qed_val, 4),
            "lipinski_violations": ro5_viol,
            "oral_veber_pass": veber_ok,
            "bbb_permeable_est": bbb_ok,
            "pains_alerts": int(pains_hits),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def encode_smiles_multimodal(
    smiles_list: list,
    chemberta_dir: str,
    morgan_radius: int = 2,
    morgan_bits: int = 1024,
    device_str: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    为任意新分子 SMILES 列表计算 RDKit Morgan 指纹与 ChemBERTa 语义嵌入。
    默认可在 CPU 运行，不抢占主训练 GPU 显存。
    """
    device = torch.device(device_str)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=int(morgan_radius), fpSize=int(morgan_bits))

    morgan_list = []
    valid_smiles = []
    for smi in smiles_list:
        smi_s = "" if smi is None else str(smi).strip()
        mol = Chem.MolFromSmiles(smi_s) if smi_s else None
        if mol is not None:
            morgan_list.append(mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32))
            valid_smiles.append(smi_s)
        else:
            morgan_list.append(np.zeros(int(morgan_bits), dtype=np.float32))
            valid_smiles.append("C")
    morgan_mat = np.vstack(morgan_list)

    tokenizer = AutoTokenizer.from_pretrained(chemberta_dir)
    model = AutoModel.from_pretrained(chemberta_dir).to(device)
    model.eval()
    cb_list = []
    for i in range(0, len(valid_smiles), 32):
        batch = valid_smiles[i : i + 32]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        cb_list.append(emb.cpu().numpy().astype(np.float32))
    chemberta_mat = np.vstack(cb_list)
    return morgan_mat, chemberta_mat


def map_molecules_to_targets_and_pathways(
    query_smiles: list,
    query_names: list,
    ref_smiles_csv: str,
    ref_mol_npz: str,
    priors_dir: str,
    chemberta_dir: str,
    top_k_ref: int = 3,
    device_str: str = "cpu",
) -> pd.DataFrame:
    """
    将任意候选分子 SMILES 通过结构/语义双模态检索映射到参考药物靶点，并关联至全量人类通路底座。
    """
    ref_df = pd.read_csv(ref_smiles_csv)
    ref_npz = np.load(ref_mol_npz, allow_pickle=True)
    ref_drugs = [str(x) for x in ref_npz["drugs"]]
    ref_morgan = ref_npz["morgan"].astype(np.float32)
    ref_cb = ref_npz["chemberta"].astype(np.float32)

    # 归一化参考指纹用于快速余弦/Tanimoto融合检索
    ref_m_norm = ref_morgan / np.maximum(np.linalg.norm(ref_morgan, axis=1, keepdims=True), 1e-6)
    ref_cb_norm = ref_cb / np.maximum(np.linalg.norm(ref_cb, axis=1, keepdims=True), 1e-6)

    q_morgan, q_cb = encode_smiles_multimodal(query_smiles, chemberta_dir=chemberta_dir, device_str=device_str)
    q_m_norm = q_morgan / np.maximum(np.linalg.norm(q_morgan, axis=1, keepdims=True), 1e-6)
    q_cb_norm = q_cb / np.maximum(np.linalg.norm(q_cb, axis=1, keepdims=True), 1e-6)

    sim_mat = 0.5 * (q_m_norm @ ref_m_norm.T) + 0.5 * (q_cb_norm @ ref_cb_norm.T)

    drug_to_target = dict(zip(ref_df["perturbation"].astype(str), ref_df["target"].fillna("").astype(str)))
    drug_to_pw1 = dict(zip(ref_df["perturbation"].astype(str), ref_df["pathway_level_1"].fillna("").astype(str)))

    mem_path = os.path.join(priors_dir, "gene_pathway_membership.parquet")
    mem_df = pd.read_parquet(mem_path)
    gene_to_pws = mem_df.groupby("gene_symbol")["pathway_name"].apply(lambda s: list(dict.fromkeys(s))[:5]).to_dict()

    rows = []
    for qi, qname in enumerate(query_names):
        sims = sim_mat[qi]
        top_idx = np.argsort(-sims)[: int(top_k_ref)]
        matched_refs = []
        cand_targets = []
        cand_pws = []
        for idx in top_idx:
            rd = ref_drugs[idx]
            sc_val = float(sims[idx])
            tgt = drug_to_target.get(rd, "")
            pw1 = drug_to_pw1.get(rd, "")
            matched_refs.append(f"{rd}({sc_val:.3f})")
            for t in [x.strip().upper() for x in tgt.split(",") if x.strip()]:
                if t not in cand_targets:
                    cand_targets.append(t)
                for p in gene_to_pws.get(t, []):
                    if p not in cand_pws:
                        cand_pws.append(p)
            if pw1 and pw1 not in cand_pws:
                cand_pws.append(pw1)

        rows.append({
            "drug_name": qname,
            "top_similar_ref_drugs": "; ".join(matched_refs),
            "inferred_targets": "; ".join(cand_targets[:6]),
            "mapped_prior_pathways": "; ".join(cand_pws[:6]),
            "max_ref_similarity": round(float(sims[top_idx[0]]), 4),
        })

    return pd.DataFrame(rows)

# 中文注释：本模块实现 AI 制药化学前端接口，可将任意新分子 SMILES 转化为 RDKit ADMET/PAINS 成药性指标，
# 并通过 Morgan+ChemBERTa 双模态检索将其映射到潜在蛋白靶点与全量人类通路先验底座。
