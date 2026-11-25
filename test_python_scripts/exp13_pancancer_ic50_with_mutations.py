# -*- coding: utf-8 -*-
"""
思路 C (Exp13): 接入 CCLE 体细胞驱动突变特征 (1,723 个突变基因: BRAF/KRAS/EGFR/TP53/PIK3CA...)
构建 Multi-Omics PanCancer-BioMolKG (RNA-seq + Somatic Mutation + Drug SMILES/ESM-2)
在 GDSC1+GDSC2 45万对泛癌数据上对比 Expr-Only vs Expr+Mutation 对 Unseen Cell Lines 及靶向药精确医学的提升！
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_ccle_mutations(mut_xlsx: str, model_csv: str) -> tuple:
    print(f"[Mutation] 读取 CCLE 体细胞突变矩阵: {mut_xlsx}")
    df = pd.read_excel(mut_xlsx)
    mut_genes = df["Unnamed: 0"].astype(str).tolist()
    mat = np.nan_to_num(df.iloc[:, 1:].fillna(0.0).values.T.astype(np.float32), nan=0.0)  # [1030, 1723]
    cl_cols = list(df.columns[1:])

    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    ml = pd.read_csv(model_csv, usecols=["model_id", "model_name"]).dropna()
    name_to_sidm = {norm(r["model_name"]): str(r["model_id"]) for _, r in ml.iterrows()}

    sidm_to_mut = {}
    for i, c in enumerate(cl_cols):
        sidm = name_to_sidm.get(norm(c))
        if sidm:
            sidm_to_mut[sidm] = mat[i]
    print(f"[Mutation] 成功映射细胞系体细胞突变谱: {len(sidm_to_mut)} 株细胞系 × {len(mut_genes)} 突变基因")
    return sidm_to_mut, mut_genes


def load_expr(tpm_csv: str, hvg_genes: list) -> dict:
    df = pd.read_csv(tpm_csv, skiprows=[1, 2, 3], low_memory=False)
    df = df.rename(columns={"model_id": "gene_symbol"}).drop(columns=["Unnamed: 1", "Unnamed: 2"], errors="ignore")
    df = df.dropna(subset=["gene_symbol"]).drop_duplicates("gene_symbol").set_index("gene_symbol")
    aligned = df.reindex(hvg_genes).fillna(0.0).astype(np.float32)
    mat = np.log1p(np.clip(aligned.values.T, 0.0, None))
    mat = (mat - mat.mean(axis=1, keepdims=True)) / np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
    return {m: mat[i] for i, m in enumerate(aligned.columns)}


class MultiOmicsIC50Net(nn.Module):
    def __init__(self, n_genes: int, n_muts: int, drug_dim: int, hidden_dim: int = 256, use_mut: bool = True):
        super().__init__()
        self.use_mut = use_mut
        self.expr_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        if use_mut:
            self.mut_enc = nn.Sequential(
                nn.Linear(n_muts, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.20),
            )
            in_c = hidden_dim + 128
        else:
            in_c = hidden_dim
        self.cell_proj = nn.Sequential(
            nn.Linear(in_c, hidden_dim),
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

    def forward(self, xc: torch.Tensor, xm: torch.Tensor, xd: torch.Tensor) -> torch.Tensor:
        he = self.expr_enc(xc)
        if self.use_mut:
            hc = self.cell_proj(torch.cat([he, self.mut_enc(xm)], dim=-1))
        else:
            hc = self.cell_proj(he)
        hd = self.drug_enc(xd)
        return self.head(torch.cat([hc, hd, hc * hd], dim=-1)).squeeze(-1)


def run_exp13():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)

    cell_expr = load_expr(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", hvg_genes)
    sidm_to_mut, mut_genes = load_ccle_mutations(
        r"D:\data\drug\DataFiles\DataFiles\GLDS\CTRPv2\ccle mut data.xlsx",
        r"D:\data\AIVC\data\cmp_model_list.csv",
    )

    d = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=False)
    drug_dict = {str(k): d["feats"][i] for i, k in enumerate(d["drugs"])}
    drug_dim = len(next(iter(drug_dict.values())))

    g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    g1 = pd.read_excel(r"D:\data\AIVC\data\GDSC1_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    gdsc = pd.concat([g2, g1], ignore_index=True).dropna(subset=["LN_IC50", "DRUG_NAME"])
    gdsc = gdsc[gdsc["SANGER_MODEL_ID"].isin(cell_expr) & gdsc["SANGER_MODEL_ID"].isin(sidm_to_mut) & gdsc["DRUG_NAME"].isin(drug_dict)]
    gdsc = gdsc.drop_duplicates(subset=["SANGER_MODEL_ID", "DRUG_NAME"], keep="first").reset_index(drop=True)
    print(f"[Multi-Omics GDSC] 共有 RNA+Mutation+Drug 有效样本: {len(gdsc)} 对 ({gdsc['SANGER_MODEL_ID'].nunique()} 株细胞系 × {gdsc['DRUG_NAME'].nunique()} 药物)")

    # 80% / 20% Unseen Cell Lines 划分
    cls = sorted(gdsc["SANGER_MODEL_ID"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(cls)
    tr_cls = set(cls[: int(0.8 * len(cls))])
    tr_df = gdsc[gdsc["SANGER_MODEL_ID"].isin(tr_cls)].reset_index(drop=True)
    te_df = gdsc[~gdsc["SANGER_MODEL_ID"].isin(tr_cls)].reset_index(drop=True)

    def pack(sub):
        xc = np.vstack([cell_expr[m] for m in sub["SANGER_MODEL_ID"]]).astype(np.float32)
        xm = np.vstack([sidm_to_mut[m] for m in sub["SANGER_MODEL_ID"]]).astype(np.float32)
        xd = np.vstack([drug_dict[dr] for dr in sub["DRUG_NAME"]]).astype(np.float32)
        y = sub["LN_IC50"].values.astype(np.float32)
        return xc, xm, xd, y

    tr_xc, tr_xm, tr_xd, tr_y = pack(tr_df)
    te_xc, te_xm, te_xd, te_y = pack(te_df)

    # 索引 BRAF_MUT, EGFR_MUT, KRAS_MUT 用于精准医学分层对比
    braf_idx = mut_genes.index("BRAF_MUT")
    egfr_idx = mut_genes.index("EGFR_MUT")
    te_braf = te_xm[:, braf_idx] > 0
    te_egfr = te_xm[:, egfr_idx] > 0

    rows = []
    for use_mut, label in [(False, "Expr-Only"), (True, "MultiOmics(Expr+CCLE_Mut)")]:
        torch.manual_seed(42)
        net = MultiOmicsIC50Net(len(hvg_genes), len(mut_genes), drug_dim, hidden_dim=256, use_mut=use_mut).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

        t_xc = torch.tensor(tr_xc, device=device)
        t_xm = torch.tensor(tr_xm, device=device)
        t_xd = torch.tensor(tr_xd, device=device)
        t_y = torch.tensor(tr_y, device=device)

        net.train()
        for ep in range(25):
            perm = torch.randperm(len(tr_y), device=device)
            for i in range(0, len(tr_y), 1024):
                idx = perm[i : i + 1024]
                opt.zero_grad()
                loss = F.mse_loss(net(t_xc[idx], t_xm[idx], t_xd[idx]), t_y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            sched.step()

        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(te_y), 2048):
                preds.append(net(
                    torch.tensor(te_xc[i : i + 2048], device=device),
                    torch.tensor(te_xm[i : i + 2048], device=device),
                    torch.tensor(te_xd[i : i + 2048], device=device),
                ).cpu().numpy())
        pred = np.concatenate(preds)

        r_all, _ = pearsonr(te_y, pred)
        sp_all, _ = spearmanr(te_y, pred)
        rmse_all = float(np.sqrt(np.mean((te_y - pred) ** 2)))

        # BRAF 靶向药 (Dabrafenib / PLX-4720 / Vemurafenib) 在 Unseen Cell Lines 上的敏感性区分度
        braf_drugs = te_df["DRUG_NAME"].isin(["Dabrafenib", "PLX-4720", "Vemurafenib"]).values
        r_braf, _ = pearsonr(te_y[braf_drugs], pred[braf_drugs]) if braf_drugs.sum() > 10 else (0.0, 1.0)
        # 预测 BRAF 突变 vs 野生型细胞对 BRAF 抑制剂的 IC50 差值 (True vs Pred)
        true_gap = float(te_y[braf_drugs & (~te_braf)].mean() - te_y[braf_drugs & te_braf].mean())
        pred_gap = float(pred[braf_drugs & (~te_braf)].mean() - pred[braf_drugs & te_braf].mean())

        print(f"  [{label}] All Unseen: Pearson={r_all:.4f} | Spearman={sp_all:.4f} | RMSE={rmse_all:.4f} || BRAF-Inhibitors r={r_braf:.4f} (WT-Mut IC50 Gap: True={true_gap:.2f}, Pred={pred_gap:.2f})")
        rows.append({
            "model": label, "n_train": len(tr_y), "n_test": len(te_y),
            "pearson_all": r_all, "spearman_all": sp_all, "rmse_all": rmse_all,
            "braf_inhibitor_pearson": r_braf, "braf_wt_mut_true_gap": true_gap, "braf_wt_mut_pred_gap": pred_gap,
        })

    pd.DataFrame(rows).to_csv(r"D:\data\AIVC\test_results\exp13_multiomics_mut_ic50.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    run_exp13()
