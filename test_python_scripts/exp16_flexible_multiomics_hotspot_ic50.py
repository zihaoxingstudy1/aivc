# -*- coding: utf-8 -*-
"""
Exp16: 模态弹性多组学泛癌药敏模型 (Modality-Flexible Multi-Omics: RNA-seq + 165 Hotspot Driver Mutations)
核心设计 ("给什么数据就用什么数据"):
- 主干永远以 RNA-seq 表达为基础；
- 仅接入 DepMap 官方精选的 165 个临床 Hotspot 驱动突变 (彻底剔除 1,600+ 乘客突变噪声)；
- 训练时使用 Modality Dropout (随机置零突变分支)，使模型在测试时既支持 (Expr + Hotspot Mut)，也支持在缺失突变数据时无缝退化为 (Expr-Only) 稳定预测！
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_hotspot_mutations(hotspot_csv: str, model_csv: str) -> tuple:
    print(f"[Hotspot] 读取 DepMap 精选驱动突变矩阵: {hotspot_csv}")
    df = pd.read_csv(hotspot_csv)
    if "IsDefaultEntryForModel" in df.columns:
        df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
    meta_cols = [c for c in ["Unnamed: 0", "SequencingID", "ModelID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC"] if c in df.columns]
    mut_cols = [c for c in df.columns if c not in meta_cols]
    mat = np.nan_to_num(df[mut_cols].fillna(0.0).values.astype(np.float32), nan=0.0)
    mat = (mat > 0).astype(np.float32)

    ml = pd.read_csv(model_csv, usecols=["model_id", "BROAD_ID"]).dropna()
    broad_to_sidm = dict(zip(ml["BROAD_ID"], ml["model_id"]))
    sidm_to_mut = {}
    for i, mid in enumerate(df["ModelID"].astype(str)):
        sidm = broad_to_sidm.get(mid)
        if sidm:
            sidm_to_mut[sidm] = mat[i]
    print(f"[Hotspot] 映射成功: {len(sidm_to_mut)} 株细胞系 × {len(mut_cols)} 个核心驱动突变基因")
    return sidm_to_mut, mut_cols


def load_expr(tpm_csv: str, hvg_genes: list) -> dict:
    df = pd.read_csv(tpm_csv, skiprows=[1, 2, 3], low_memory=False)
    df = df.rename(columns={"model_id": "gene_symbol"}).drop(columns=["Unnamed: 1", "Unnamed: 2"], errors="ignore")
    df = df.dropna(subset=["gene_symbol"]).drop_duplicates("gene_symbol").set_index("gene_symbol")
    aligned = df.reindex(hvg_genes).fillna(0.0).astype(np.float32)
    mat = np.log1p(np.clip(aligned.values.T, 0.0, None))
    mat = (mat - mat.mean(axis=1, keepdims=True)) / np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
    return {m: mat[i] for i, m in enumerate(aligned.columns)}


class FlexibleMultiOmicsIC50Net(nn.Module):
    """弹性多模态交互网络: 有突变时融合门控残差，无突变时自动退化为纯转录组。"""
    def __init__(self, n_genes: int, n_muts: int, drug_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.expr_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mut_enc = nn.Sequential(
            nn.Linear(n_muts, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(64, hidden_dim),
        )
        self.mut_gate = nn.Parameter(torch.tensor(0.2))
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

    def forward(self, xc: torch.Tensor, xm: torch.Tensor, xd: torch.Tensor, mut_mask: torch.Tensor) -> torch.Tensor:
        hc = self.expr_enc(xc)
        hm = self.mut_enc(xm) * mut_mask.unsqueeze(-1)
        hc = hc + torch.tanh(self.mut_gate) * hm
        hd = self.drug_enc(xd)
        return self.head(torch.cat([hc, hd, hc * hd], dim=-1)).squeeze(-1)


def run_exp16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)

    cell_expr = load_expr(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", hvg_genes)
    sidm_to_mut, mut_cols = load_hotspot_mutations(
        r"D:\data\drug\DeepMapcrisp\OmicsSomaticMutationsMatrixHotspot.csv",
        r"D:\data\AIVC\data\cmp_model_list.csv",
    )

    d = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=False)
    drug_dict = {str(k): d["feats"][i] for i, k in enumerate(d["drugs"])}
    drug_dim = len(next(iter(drug_dict.values())))

    g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    g1 = pd.read_excel(r"D:\data\AIVC\data\GDSC1_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    gdsc = pd.concat([g2, g1], ignore_index=True).dropna(subset=["LN_IC50", "DRUG_NAME"])
    gdsc = gdsc[gdsc["SANGER_MODEL_ID"].isin(cell_expr) & gdsc["DRUG_NAME"].isin(drug_dict)]
    gdsc = gdsc.drop_duplicates(subset=["SANGER_MODEL_ID", "DRUG_NAME"], keep="first").reset_index(drop=True)

    cls = sorted(gdsc["SANGER_MODEL_ID"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(cls)
    tr_cls = set(cls[: int(0.8 * len(cls))])
    tr_df = gdsc[gdsc["SANGER_MODEL_ID"].isin(tr_cls)].reset_index(drop=True)
    te_df = gdsc[~gdsc["SANGER_MODEL_ID"].isin(tr_cls)].reset_index(drop=True)

    n_muts = len(mut_cols)
    zero_mut = np.zeros(n_muts, dtype=np.float32)

    def pack(sub):
        xc = np.vstack([cell_expr[m] for m in sub["SANGER_MODEL_ID"]]).astype(np.float32)
        xm = np.vstack([sidm_to_mut.get(m, zero_mut) for m in sub["SANGER_MODEL_ID"]]).astype(np.float32)
        mm = np.array([1.0 if m in sidm_to_mut else 0.0 for m in sub["SANGER_MODEL_ID"]], dtype=np.float32)
        xd = np.vstack([drug_dict[dr] for dr in sub["DRUG_NAME"]]).astype(np.float32)
        y = sub["LN_IC50"].values.astype(np.float32)
        return xc, xm, mm, xd, y

    tr_xc, tr_xm, tr_mm, tr_xd, tr_y = pack(tr_df)
    te_xc, te_xm, te_mm, te_xd, te_y = pack(te_df)
    print(f"[Exp16] 训练={len(tr_y)} (含突变={int(tr_mm.sum())}) | 测试={len(te_y)} (含突变={int(te_mm.sum())})")

    braf_col = next((i for i, c in enumerate(mut_cols) if c.startswith("BRAF ")), None)
    braf_drugs = te_df["DRUG_NAME"].isin(["Dabrafenib", "PLX-4720", "Vemurafenib"]).values

    torch.manual_seed(42)
    net = FlexibleMultiOmicsIC50Net(len(hvg_genes), n_muts, drug_dim, hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    t_xc = torch.tensor(tr_xc, device=device)
    t_xm = torch.tensor(tr_xm, device=device)
    t_mm = torch.tensor(tr_mm, device=device)
    t_xd = torch.tensor(tr_xd, device=device)
    t_y = torch.tensor(tr_y, device=device)

    net.train()
    for ep in range(25):
        perm = torch.randperm(len(tr_y), device=device)
        for i in range(0, len(tr_y), 1024):
            idx = perm[i : i + 1024]
            # Modality Dropout: 30% 概率随机关闭突变分支，迫使模型同时学会纯转录组与融合突变
            drop = (torch.rand(len(idx), device=device) > 0.30).float()
            mm_batch = t_mm[idx] * drop
            opt.zero_grad()
            loss = F.mse_loss(net(t_xc[idx], t_xm[idx], t_xd[idx], mm_batch), t_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sched.step()

    net.eval()
    rows = []
    for mode, use_real_mask in [("FlexibleModel_TestWithHotspotMut", True), ("FlexibleModel_TestWithoutMut(Fallback)", False)]:
        preds = []
        with torch.no_grad():
            for i in range(0, len(te_y), 2048):
                mm_b = torch.tensor(te_mm[i : i + 2048], device=device) if use_real_mask else torch.zeros(min(2048, len(te_y) - i), device=device)
                preds.append(net(
                    torch.tensor(te_xc[i : i + 2048], device=device),
                    torch.tensor(te_xm[i : i + 2048], device=device),
                    torch.tensor(te_xd[i : i + 2048], device=device),
                    mm_b,
                ).cpu().numpy())
        pred = np.concatenate(preds)
        r_all, _ = pearsonr(te_y, pred)
        sp_all, _ = spearmanr(te_y, pred)
        rmse = float(np.sqrt(np.mean((te_y - pred) ** 2)))
        r_braf, _ = pearsonr(te_y[braf_drugs], pred[braf_drugs]) if braf_drugs.sum() > 10 else (0.0, 1.0)
        print(f"  [{mode}] Pearson={r_all:.4f} | Spearman={sp_all:.4f} | RMSE={rmse:.4f} | BRAF-Inhibitors r={r_braf:.4f}")
        rows.append({"mode": mode, "pearson": r_all, "spearman": sp_all, "rmse": rmse, "braf_pearson": r_braf})

    pd.DataFrame(rows).to_csv(r"D:\data\AIVC\test_results\exp16_flexible_hotspot_ic50.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    run_exp16()
