# -*- coding: utf-8 -*-
"""
Exp17: 接入 DepMap PRISM 20Q2 Secondary (73.9万条 × 1,488种临床/非肿瘤老药) + DepMap CRISPRGeneEffect (18,434基因致死率)
1. PRISM Drug Repurposing (AUC 预测): Unseen Cell Lines & Unseen Drugs Zero-Shot 评测 (用自带 SMILES + ChemBERTa + Morgan + ESM2 靶点)
2. CRISPR Gene Lethality (基因敲除致死率 Chronos Score 预测): Unseen Cell Lines & Unseen Target Genes Zero-Shot 评测 (用 RNA-seq + ESM2 靶点蛋白嵌入)
彻底打通“老药新用 (Repurposing)”与“合成致死靶点发现 (CRISPR Lethality)”双闭环！
"""

import os
import sys
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


def load_depmap_expr(expr_csv: str, hvg_genes: list) -> dict:
    print(f"[DepMap-Expr] 读取 {expr_csv} ...")
    df = pd.read_csv(expr_csv, low_memory=False)
    if "IsDefaultEntryForModel" in df.columns:
        df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
    id_col = "ModelID" if "ModelID" in df.columns else df.columns[0]
    # 列名格式如 "TSPAN6 (7105)" -> 提取基因名
    meta = {"Unnamed: 0", "SequencingID", "ModelID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC"}
    col_map = {}
    for c in df.columns:
        if c in meta:
            continue
        sym = str(c).split(" (")[0].strip()
        if sym not in col_map:
            col_map[sym] = c
    keep_cols = [col_map[g] for g in hvg_genes if g in col_map]
    keep_syms = [g for g in hvg_genes if g in col_map]
    sub = df[keep_cols].fillna(0.0).values.astype(np.float32)
    sub = (sub - sub.mean(axis=1, keepdims=True)) / np.maximum(sub.std(axis=1, keepdims=True), 1e-6)

    # 补全到完整 hvg_genes 维度
    sym_idx = {g: i for i, g in enumerate(keep_syms)}
    full = np.zeros((len(df), len(hvg_genes)), dtype=np.float32)
    for gi, g in enumerate(hvg_genes):
        if g in sym_idx:
            full[:, gi] = sub[:, sym_idx[g]]
    mids = df[id_col].astype(str).tolist()
    print(f"[DepMap-Expr] 对齐完成: {len(mids)} 株细胞系 × {len(keep_syms)}/{len(hvg_genes)} 共有基因")
    return {m: full[i] for i, m in enumerate(mids)}


@torch.no_grad()
def encode_prism_smiles(drug_smiles: dict, cache_npz: str) -> dict:
    if os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=False)
        return {str(k): d["feats"][i] for i, k in enumerate(d["drugs"])}

    drugs = sorted(drug_smiles.keys())
    print(f"[PRISM-Mol] 编码 {len(drugs)} 种药物的 Morgan + ChemBERTa 特征 ...")
    morgan_list = []
    for dr in drugs:
        sm = str(drug_smiles[dr]).split(",")[0].strip()
        mol = Chem.MolFromSmiles(sm) if sm else None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32) if mol is not None else np.zeros(1024, dtype=np.float32)
        morgan_list.append(fp)
    morgan_mat = np.vstack(morgan_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(r"D:\data\AIVC\pretrained\chemberta")
    cb = AutoModel.from_pretrained(r"D:\data\AIVC\pretrained\chemberta").to(device).eval()
    cb_list = []
    for i in range(0, len(drugs), 64):
        sms = [str(drug_smiles[dr]).split(",")[0].strip() or "C" for dr in drugs[i : i + 64]]
        enc = tok(sms, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = cb(**enc).last_hidden_state[:, 0, :]
        cb_list.append(F.normalize(out, dim=-1).cpu().numpy().astype(np.float32))
    cb_mat = np.vstack(cb_list)
    del cb
    torch.cuda.empty_cache()

    feats = np.hstack([morgan_mat, cb_mat]).astype(np.float32)  # 1024 + 384 = 1408
    np.savez(cache_npz, drugs=np.array(drugs), feats=feats)
    return {dr: feats[i] for i, dr in enumerate(drugs)}


class BilinearInteractionNet(nn.Module):
    def __init__(self, cell_dim: int, target_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.tgt_enc = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
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

    def forward(self, xc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        hc = self.cell_enc(xc)
        ht = self.tgt_enc(xt)
        return self.head(torch.cat([hc, ht, hc * ht], dim=-1)).squeeze(-1)


def eval_pair_task(name: str, tr_xc, tr_xt, tr_y, te_xc, te_xt, te_y, device, epochs=18):
    # Ridge 基线
    ridge = Ridge(alpha=500.0).fit(np.hstack([tr_xc, tr_xt]), tr_y)
    pred_r = ridge.predict(np.hstack([te_xc, te_xt]))
    r_ridge, _ = pearsonr(te_y, pred_r)
    sp_ridge, _ = spearmanr(te_y, pred_r)

    torch.manual_seed(42)
    net = BilinearInteractionNet(tr_xc.shape[1], tr_xt.shape[1], hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t_xc = torch.tensor(tr_xc, device=device)
    t_xt = torch.tensor(tr_xt, device=device)
    t_y = torch.tensor(tr_y, device=device)

    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr_y), device=device)
        for i in range(0, len(tr_y), 1024):
            idx = perm[i : i + 1024]
            opt.zero_grad()
            loss = F.mse_loss(net(t_xc[idx], t_xt[idx]), t_y[idx])
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_y), 2048):
            preds.append(net(
                torch.tensor(te_xc[i : i + 2048], device=device),
                torch.tensor(te_xt[i : i + 2048], device=device),
            ).cpu().numpy())
    pred_nn = np.concatenate(preds)
    r_nn, _ = pearsonr(te_y, pred_nn)
    sp_nn, _ = spearmanr(te_y, pred_nn)
    rmse_nn = float(np.sqrt(np.mean((te_y - pred_nn) ** 2)))
    print(f"  [{name}] Train={len(tr_y)} | Test={len(te_y)}")
    print(f"    Linear-Ridge: Pearson={r_ridge:.4f} | Spearman={sp_ridge:.4f}")
    print(f"    BioMolKG-Net: Pearson={r_nn:.4f} | Spearman={sp_nn:.4f} | RMSE={rmse_nn:.4f}")
    return {"task": name, "n_train": len(tr_y), "n_test": len(te_y), "ridge_pearson": r_ridge, "ridge_spearman": sp_ridge, "nn_pearson": r_nn, "nn_spearman": sp_nn, "nn_rmse": rmse_nn}


def run_exp17():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)

    cell_expr = load_depmap_expr(r"D:\data\drug\DeepMapcrisp\OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", hvg_genes)

    # 1. PRISM 20Q2 Secondary (老药新用 AUC)
    print("\n=== Benchmark 1: DepMap PRISM 20Q2 Secondary (73.9万条 × 1,488 种药物) ===")
    p2 = pd.read_csv(
        r"D:\data\drug\CCLE_drug\prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv",
        usecols=["depmap_id", "name", "auc", "smiles"],
        low_memory=False,
    ).dropna(subset=["depmap_id", "name", "auc", "smiles"])
    p2 = p2[p2["depmap_id"].isin(cell_expr) & p2["auc"].between(0.0, 1.5)]
    p2 = p2.groupby(["depmap_id", "name"], as_index=False).agg({"auc": "mean", "smiles": "first"})
    print(f"[PRISM] 有效匹配样本: {len(p2)} 对 ({p2['depmap_id'].nunique()} 株细胞系 × {p2['name'].nunique()} 种药物)")

    d_sm = dict(zip(p2["name"], p2["smiles"]))
    prism_mol = encode_prism_smiles(d_sm, r"D:\data\AIVC\data\prism_20q2_mol_features.npz")

    rng = np.random.default_rng(42)
    cls = sorted(p2["depmap_id"].unique())
    rng.shuffle(cls)
    tr_cls = set(cls[: int(0.8 * len(cls))])

    drs = sorted(p2["name"].unique())
    rng.shuffle(drs)
    tr_drs = set(drs[: int(0.8 * len(drs))])

    def pack_prism(sub):
        return (
            np.vstack([cell_expr[m] for m in sub["depmap_id"]]).astype(np.float32),
            np.vstack([prism_mol[d] for d in sub["name"]]).astype(np.float32),
            sub["auc"].values.astype(np.float32),
        )

    res = []
    tr_a, te_a = p2[p2["depmap_id"].isin(tr_cls)], p2[~p2["depmap_id"].isin(tr_cls)]
    res.append(eval_pair_task("PRISM_UnseenCellLines", *pack_prism(tr_a), *pack_prism(te_a), device))

    tr_b, te_b = p2[p2["name"].isin(tr_drs)], p2[~p2["name"].isin(tr_drs)]
    res.append(eval_pair_task("PRISM_UnseenDrugs_ZeroShot", *pack_prism(tr_b), *pack_prism(te_b), device))

    # 2. DepMap CRISPRGeneEffect (基因敲除致死率 Chronos Score)
    print("\n=== Benchmark 2: DepMap CRISPRGeneEffect (细胞致死率预测) ===")
    esm_dict = {}
    for npz_p in (r"D:\data\AIVC\data\target_esm2_150m.npz", r"D:\data\AIVC\data\unified_esm2_targets.npz"):
        if os.path.exists(npz_p):
            esm_d = np.load(npz_p, allow_pickle=False)
            kname = "genes" if "genes" in esm_d else "targets"
            vname = "embeddings" if "embeddings" in esm_d else ("embs" if "embs" in esm_d else "target_embs")
            for i, k in enumerate(esm_d[kname]):
                if np.linalg.norm(esm_d[vname][i]) > 1e-6:
                    esm_dict[str(k).upper()] = esm_d[vname][i]

    cr = pd.read_csv(r"D:\data\drug\DeepMapcrisp\CRISPRGeneEffect.csv", low_memory=False)
    id_col = cr.columns[0]
    cr = cr[cr[id_col].astype(str).isin(cell_expr)].copy()
    gene_cols = [c for c in cr.columns[1:] if str(c).split(" (")[0].strip().upper() in esm_dict]
    # 筛选方差最大的 200 个核心靶点基因 (涵盖必需基因与选择性依赖基因)
    stds = cr[gene_cols].std().sort_values(ascending=False)
    sel_cols = list(stds.head(200).index)
    sel_syms = [str(c).split(" (")[0].strip().upper() for c in sel_cols]
    print(f"[CRISPRGeneEffect] 对齐 {len(cr)} 株细胞系 × Top {len(sel_cols)} 高方差靶点基因 (总计 {len(cr)*len(sel_cols)} 对)")

    # 展平为 (Cell, Gene) 对
    cr_mids = cr[id_col].astype(str).tolist()
    cr_mat = cr[sel_cols].fillna(0.0).values.astype(np.float32)

    rng2 = np.random.default_rng(42)
    c_idx = np.arange(len(cr_mids))
    rng2.shuffle(c_idx)
    tr_c, te_c = c_idx[: int(0.8 * len(c_idx))], c_idx[int(0.8 * len(c_idx)) :]

    def build_cr_pairs(row_ids, col_ids):
        xc_l, xt_l, y_l = [], [], []
        for ri in row_ids:
            cv = cell_expr[cr_mids[ri]]
            for ci in col_ids:
                xc_l.append(cv)
                xt_l.append(esm_dict[sel_syms[ci]])
                y_l.append(cr_mat[ri, ci])
        return np.vstack(xc_l).astype(np.float32), np.vstack(xt_l).astype(np.float32), np.array(y_l, dtype=np.float32)

    all_g_idx = np.arange(len(sel_cols))
    res.append(eval_pair_task("CRISPR_Lethality_UnseenCellLines", *build_cr_pairs(tr_c, all_g_idx), *build_cr_pairs(te_c, all_g_idx), device, epochs=15))

    rng2.shuffle(all_g_idx)
    tr_g, te_g = all_g_idx[: int(0.8 * len(all_g_idx))], all_g_idx[int(0.8 * len(all_g_idx)) :]
    res.append(eval_pair_task("CRISPR_Lethality_UnseenTargets", *build_cr_pairs(c_idx, tr_g), *build_cr_pairs(c_idx, te_g), device, epochs=15))

    pd.DataFrame(res).to_csv(r"D:\data\AIVC\test_results\exp17_prism_and_crispr_lethality.csv", index=False, encoding="utf-8")
    print("\n[Exp17] 结果已保存至 exp17_prism_and_crispr_lethality.csv")


if __name__ == "__main__":
    run_exp17()
