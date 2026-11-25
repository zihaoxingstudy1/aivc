# -*- coding: utf-8 -*-
"""
方向 5 终极规模化 (Exp9): 泛癌 GDSC1 + GDSC2 + Broad/Sanger CCLE (1,898 株癌细胞系 × 数百种小分子) IC50 联合预测
严格双维度 OOD 评测：
1. Split A (Unseen Cancer Cell Lines): 80% 细胞系训练 -> 20% 未见癌细胞系测试
2. Split B (Unseen Drugs Zero-Shot): 80% 药物训练 -> 20% 全新未见药物测试 (靠 SMILES + ESM-2 靶点泛化)
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_ccle_sanger_expr(tpm_csv: str, ref_hvg_genes: list) -> tuple:
    print(f"[CCLE/Sanger] 读取 {tpm_csv} ...")
    df = pd.read_csv(tpm_csv, skiprows=[1, 2, 3], low_memory=False)
    df = df.rename(columns={"model_id": "gene_symbol"}).drop(columns=["Unnamed: 1", "Unnamed: 2"], errors="ignore")
    df = df.dropna(subset=["gene_symbol"]).drop_duplicates("gene_symbol").set_index("gene_symbol")
    aligned = df.reindex(ref_hvg_genes).fillna(0.0).astype(np.float32)
    mat = np.log1p(np.clip(aligned.values.T, 0.0, None))  # [N_models, N_genes]
    # 标准化每个细胞系表达谱 (Zero-mean unit-norm / Z-score per sample) 以对齐跨平台基态
    mu = mat.mean(axis=1, keepdims=True)
    sd = np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
    mat = (mat - mu) / sd
    model_ids = list(aligned.columns)
    print(f"[CCLE/Sanger] 对齐完成: {len(model_ids)} 株细胞系 × {mat.shape[1]} 基因")
    return {m: mat[i] for i, m in enumerate(model_ids)}, mat.shape[1]


def _fetch_pubchem_smiles(drug_name: str) -> tuple:
    clean = re.sub(r"\s*\(.*?\)", "", str(drug_name)).strip()
    for q in (clean, str(drug_name).strip()):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(q)}/property/CanonicalSMILES,IsomericSMILES/JSON"
        try:
            res = json.loads(urllib.request.urlopen(url, timeout=8).read().decode("utf-8"))
            props = res["PropertyTable"]["Properties"][0]
            sm = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES", "")
            if sm:
                return drug_name, sm
        except Exception:
            continue
    return drug_name, ""


def _fetch_uniprot_seq(target_kw: str) -> tuple:
    clean_tg = re.sub(r"\(.*?\)", "", str(target_kw)).split(",")[0].strip()
    if not clean_tg or clean_tg.lower() in ("nan", "others", "other", "unclassified", "dna", "microtubules"):
        return target_kw, ""
    q = f"({clean_tg}) AND organism_id:9606 AND reviewed:true"
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({"query": q, "fields": "sequence", "format": "json", "size": 1})
    try:
        res = json.loads(urllib.request.urlopen(url, timeout=8).read().decode("utf-8")).get("results", [])
        if res:
            return target_kw, res[0].get("sequence", {}).get("value", "")[:1022]
    except Exception:
        pass
    return target_kw, ""


@torch.no_grad()
def build_gdsc_drug_features(drugs: list, drug_to_target: dict, cache_npz: str) -> dict:
    if os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=False)
        return {str(k): d["feats"][i] for i, k in enumerate(d["drugs"])}

    print(f"[DrugKG] 并发查询 PubChem SMILES ({len(drugs)} 种药物) ...")
    smiles_map = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_fetch_pubchem_smiles, dr) for dr in drugs]
        for fut in as_completed(futs):
            dr, sm = fut.result()
            smiles_map[dr] = sm
    print(f"[DrugKG] PubChem 命中 SMILES: {sum(bool(v) for v in smiles_map.values())}/{len(drugs)}")

    morgan_list = []
    for dr in drugs:
        sm = smiles_map.get(dr, "")
        mol = Chem.MolFromSmiles(sm) if sm else None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32) if mol is not None else np.zeros(1024, dtype=np.float32)
        morgan_list.append(fp)
    morgan_mat = np.vstack(morgan_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb_dir = r"D:\data\AIVC\pretrained\chemberta"
    tok_cb = AutoTokenizer.from_pretrained(cb_dir)
    cb = AutoModel.from_pretrained(cb_dir).to(device).eval()
    cb_list = []
    for i in range(0, len(drugs), 32):
        batch_sm = [smiles_map.get(dr, "") or "C" for dr in drugs[i : i + 32]]
        enc = tok_cb(batch_sm, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = cb(**enc).last_hidden_state[:, 0, :]
        cb_list.append(F.normalize(out, dim=-1).cpu().numpy().astype(np.float32))
    cb_mat = np.vstack(cb_list)
    del cb

    # UniProt + ESM2 靶点蛋白嵌入
    uniq_tgs = sorted({str(drug_to_target.get(dr, "")) for dr in drugs if str(drug_to_target.get(dr, "")) not in ("", "nan")})
    print(f"[DrugKG] 并发查询 UniProt 靶点序列 ({len(uniq_tgs)} 个靶点) ...")
    tg_seq = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_fetch_uniprot_seq, tg) for tg in uniq_tgs]
        for fut in as_completed(futs):
            tg, seq = fut.result()
            if seq:
                tg_seq[tg] = seq
    print(f"[DrugKG] UniProt 成功匹配靶点序列: {len(tg_seq)}/{len(uniq_tgs)}")

    esm_dir = r"D:\data\AIVC\pretrained\esm2_150m"
    tok_e = AutoTokenizer.from_pretrained(esm_dir)
    esm = AutoModel.from_pretrained(esm_dir, torch_dtype=torch.float16).to(device).eval()
    tg_keys = sorted(tg_seq.keys())
    tg_emb = {}
    for i in range(0, len(tg_keys), 16):
        seqs = [tg_seq[k] for k in tg_keys[i : i + 16]]
        enc = tok_e(seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
        out = esm(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        e = F.normalize((out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6), dim=-1).cpu().numpy().astype(np.float32)
        for j, k in enumerate(tg_keys[i : i + 16]):
            tg_emb[k] = e[j]
    del esm
    torch.cuda.empty_cache()

    esm_mat = np.vstack([tg_emb.get(str(drug_to_target.get(dr, "")), np.zeros(640, dtype=np.float32)) for dr in drugs]).astype(np.float32)
    feats = np.hstack([morgan_mat, cb_mat, esm_mat]).astype(np.float32)
    np.savez(cache_npz, drugs=np.array(drugs), feats=feats)
    return {dr: feats[i] for i, dr in enumerate(drugs)}


class PanCancerIC50Net(nn.Module):
    """泛癌双线性+MLP交互药敏预测网络：输入 (细胞系基态表达, 药物SMILES+ESM2靶点)，预测 LN_IC50。"""
    def __init__(self, n_genes: int, drug_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, xc: torch.Tensor, xd: torch.Tensor) -> torch.Tensor:
        hc = self.cell_enc(xc)
        hd = self.drug_enc(xd)
        inter = hc * hd
        return self.head(torch.cat([hc, hd, inter], dim=-1)).squeeze(-1)


def train_and_eval_split(name: str, tr_df: pd.DataFrame, te_df: pd.DataFrame, cell_dict: dict, drug_dict: dict, n_genes: int, drug_dim: int, device):
    tr_xc = np.vstack([cell_dict[m] for m in tr_df["SANGER_MODEL_ID"]]).astype(np.float32)
    tr_xd = np.vstack([drug_dict[d] for d in tr_df["DRUG_NAME"]]).astype(np.float32)
    tr_y = tr_df["LN_IC50"].values.astype(np.float32)

    te_xc = np.vstack([cell_dict[m] for m in te_df["SANGER_MODEL_ID"]]).astype(np.float32)
    te_xd = np.vstack([drug_dict[d] for d in te_df["DRUG_NAME"]]).astype(np.float32)
    te_y = te_df["LN_IC50"].values.astype(np.float32)

    # Ridge 基线
    ridge = Ridge(alpha=500.0).fit(np.hstack([tr_xc, tr_xd]), tr_y)
    pred_r = ridge.predict(np.hstack([te_xc, te_xd]))
    r_ridge, _ = pearsonr(te_y, pred_r)
    sp_ridge, _ = spearmanr(te_y, pred_r)

    # PanCancerIC50Net
    torch.manual_seed(42)
    net = PanCancerIC50Net(n_genes, drug_dim, hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    t_xc = torch.tensor(tr_xc, device=device)
    t_xd = torch.tensor(tr_xd, device=device)
    t_y = torch.tensor(tr_y, device=device)

    net.train()
    for ep in range(25):
        perm = torch.randperm(len(tr_y), device=device)
        for i in range(0, len(tr_y), 1024):
            idx = perm[i : i + 1024]
            opt.zero_grad()
            out = net(t_xc[idx], t_xd[idx])
            loss = F.mse_loss(out, t_y[idx])
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_y), 2048):
            bx = torch.tensor(te_xc[i : i + 2048], device=device)
            bd = torch.tensor(te_xd[i : i + 2048], device=device)
            preds.append(net(bx, bd).cpu().numpy())
    pred_nn = np.concatenate(preds)
    r_nn, _ = pearsonr(te_y, pred_nn)
    sp_nn, _ = spearmanr(te_y, pred_nn)
    rmse_nn = float(np.sqrt(np.mean((te_y - pred_nn) ** 2)))

    print(f"  [{name}] Train={len(tr_y)} | Test={len(te_y)}")
    print(f"    Linear-Ridge:      Pearson={r_ridge:.4f} | Spearman={sp_ridge:.4f}")
    print(f"    PanCancer-BioMolKG: Pearson={r_nn:.4f} | Spearman={sp_nn:.4f} | RMSE={rmse_nn:.4f}")
    return {"split": name, "n_train": len(tr_y), "n_test": len(te_y), "ridge_pearson": r_ridge, "ridge_spearman": sp_ridge, "nn_pearson": r_nn, "nn_spearman": sp_nn, "nn_rmse": rmse_nn}


def run_exp9():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)

    cell_dict, n_genes = load_ccle_sanger_expr(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", hvg_genes)

    print("[GDSC1+GDSC2] 读取并合并 GDSC2 与 GDSC1 药敏数据 ...")
    g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "PUTATIVE_TARGET", "LN_IC50"])
    g1 = pd.read_excel(r"D:\data\AIVC\data\GDSC1_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "PUTATIVE_TARGET", "LN_IC50"])
    # GDSC2 优先，GDSC1 补充未覆盖的 (细胞系, 药物) 对
    gdsc = pd.concat([g2.assign(src="GDSC2"), g1.assign(src="GDSC1")], ignore_index=True)
    gdsc = gdsc[gdsc["SANGER_MODEL_ID"].isin(cell_dict)].dropna(subset=["LN_IC50", "DRUG_NAME"])
    gdsc = gdsc.drop_duplicates(subset=["SANGER_MODEL_ID", "DRUG_NAME"], keep="first").reset_index(drop=True)
    print(f"[GDSC1+GDSC2] 有效匹配总样本: {len(gdsc)} 对 ({gdsc['SANGER_MODEL_ID'].nunique()} 株癌细胞系 × {gdsc['DRUG_NAME'].nunique()} 种药物)")

    drugs = sorted(gdsc["DRUG_NAME"].unique())
    d2t = gdsc.groupby("DRUG_NAME")["PUTATIVE_TARGET"].first().to_dict()
    drug_dict = build_gdsc_drug_features(drugs, d2t, r"D:\data\AIVC\data\gdsc_drug_kg_features.npz")
    drug_dim = len(next(iter(drug_dict.values())))

    rng = np.random.default_rng(42)
    # Split A: Unseen Cancer Cell Lines
    all_cls = sorted(gdsc["SANGER_MODEL_ID"].unique())
    rng.shuffle(all_cls)
    tr_cls = set(all_cls[: int(0.8 * len(all_cls))])
    res_a = train_and_eval_split(
        "SplitA_UnseenCellLines",
        gdsc[gdsc["SANGER_MODEL_ID"].isin(tr_cls)],
        gdsc[~gdsc["SANGER_MODEL_ID"].isin(tr_cls)],
        cell_dict, drug_dict, n_genes, drug_dim, device,
    )

    # Split B: Unseen Drugs Zero-Shot
    all_drs = sorted(gdsc["DRUG_NAME"].unique())
    rng.shuffle(all_drs)
    tr_drs = set(all_drs[: int(0.8 * len(all_drs))])
    res_b = train_and_eval_split(
        "SplitB_UnseenDrugs",
        gdsc[gdsc["DRUG_NAME"].isin(tr_drs)],
        gdsc[~gdsc["DRUG_NAME"].isin(tr_drs)],
        cell_dict, drug_dict, n_genes, drug_dim, device,
    )

    res_df = pd.DataFrame([res_a, res_b])
    res_df.to_csv(r"D:\data\AIVC\test_results\exp9_pan_cancer_gdsc_ic50.csv", index=False, encoding="utf-8")
    print("\n[Exp9] 结果已保存至 exp9_pan_cancer_gdsc_ic50.csv")


if __name__ == "__main__":
    run_exp9()
