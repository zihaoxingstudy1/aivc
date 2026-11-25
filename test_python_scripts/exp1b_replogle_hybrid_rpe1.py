# -*- coding: utf-8 -*-
"""
方向 1 收官实验：Geneformer-V2-316M 语义特征增强混合岭回归 + 低秩残差校准攻克 Replogle RPE1
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings


class LowRankPriorCalibrator(nn.Module):
    """低秩瓶口残差校准器 (rank=32)：极少自由度，防止在跨细胞系上过拟合。"""
    def __init__(self, n_genes: int, prior_dim: int, rank: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(prior_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(64, rank),
        )
        self.dec = nn.Linear(rank, n_genes)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, gf_feat, delta_lin):
        return delta_lin + torch.tanh(self.alpha) * self.dec(self.enc(gf_feat))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(r"D:\data\AIVC\data\MultiLineage_Replogle_RPE1.h5ad")
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    tr_samples = []
    for cl in sorted(tr.obs["cell_line"].unique()):
        sub_cl = tr[tr.obs["cell_line"] == cl]
        cm = (sub_cl.obs["perturbation"].astype(str) == "control").values
        if cm.sum() == 0:
            continue
        mu_c = _to_dense_array(sub_cl.X[cm]).mean(axis=0)
        d_dict = _compute_group_deltas(sub_cl[~cm], "perturbation", mu_c)
        for p, dt in d_dict.items():
            tr_samples.append((cl, p, dt))

    ev_cm = (ev.obs["perturbation"].astype(str) == "control").values
    mu_rpe1 = _to_dense_array(ev.X[ev_cm]).mean(axis=0)
    ev_deltas = _compute_group_deltas(ev[~ev_cm], "perturbation", mu_rpe1)
    ev_keys = sorted(ev_deltas.keys())

    def parse_toks(ps):
        return [t.strip() for t in ps.split("_") if t.strip()]

    all_p = sorted({s[1] for s in tr_samples} | set(ev_keys))
    vocab = sorted({t for p in all_p for t in parse_toks(p)})
    tr_vocab = sorted({t for s in tr_samples for t in parse_toks(s[1])})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    var_ens = {g: str(adata.var["ensembl_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensembl_id" in adata.var.columns else None
    embs = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    gf_dim = len(next(iter(embs.values())))

    def build_hybrid_feat(plist, gf_weight=0.5):
        m_hot = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        m_gf = np.zeros((len(plist), gf_dim), dtype=np.float32)
        for ri, ps in enumerate(plist):
            toks = parse_toks(ps)
            vecs = []
            for t in toks:
                if t in lin_idx:
                    m_hot[ri, lin_idx[t]] = 1.0
                if t in embs:
                    vecs.append(embs[t])
            if vecs:
                m_gf[ri] = np.mean(vecs, axis=0) * gf_weight
        return np.hstack([m_hot, m_gf]), m_gf

    Y_tr = np.vstack([s[2] for s in tr_samples])
    X_tr_hyb, X_tr_gf = build_hybrid_feat([s[1] for s in tr_samples], gf_weight=0.6)
    X_ev_hyb, X_ev_gf = build_hybrid_feat(ev_keys, gf_weight=0.6)

    # 1. GF316M 语义增强混合岭回归
    ridge_hyb = Ridge(alpha=15.0, fit_intercept=True)
    ridge_hyb.fit(X_tr_hyb, Y_tr)
    pred_tr_hyb = ridge_hyb.predict(X_tr_hyb).astype(np.float32)
    pred_ev_hyb = ridge_hyb.predict(X_ev_hyb).astype(np.float32)

    df_hyb = evaluate_perturbation_predictions(ev_deltas, {p: pred_ev_hyb[i] for i, p in enumerate(ev_keys)}, "GF316M-Hybrid-Ridge", (20, 50))
    print(f"[Exp1B] GF316M-Hybrid-Ridge -> RPE1: Pearson={df_hyb['pearson_delta_all'].mean():.4f} | Top20={df_hyb['pearson_delta_top20'].mean():.4f} | MSE={df_hyb['mse_all'].mean():.6f}")

    # 2. 低秩残差神经校准 (rank=32)
    net = LowRankPriorCalibrator(adata.n_vars, gf_dim, rank=32).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=5e-2)
    t_gf_tr = torch.tensor(X_tr_gf, device=device)
    t_dl_tr = torch.tensor(pred_tr_hyb, device=device)
    t_y_tr = torch.tensor(Y_tr, device=device)
    t_gf_ev = torch.tensor(X_ev_gf, device=device)
    t_dl_ev = torch.tensor(pred_ev_hyb, device=device)

    for ep in range(1, 51):
        net.train()
        opt.zero_grad()
        pred = net(t_gf_tr, t_dl_tr)
        loss = F.mse_loss(pred, t_y_tr) * 50.0 + (1.0 - F.cosine_similarity(pred, t_y_tr, dim=-1)).mean() * 0.3
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred_ev_cal = net(t_gf_ev, t_dl_ev).cpu().numpy()
    df_cal = evaluate_perturbation_predictions(ev_deltas, {p: pred_ev_cal[i] for i, p in enumerate(ev_keys)}, "GF316M-LowRankCalib", (20, 50))
    print(f"[Exp1B] GF316M-LowRankCalib -> RPE1: Pearson={df_cal['pearson_delta_all'].mean():.4f} | Top20={df_cal['pearson_delta_top20'].mean():.4f} | MSE={df_cal['mse_all'].mean():.6f}")

    pd.concat([df_hyb, df_cal], ignore_index=True).to_csv(r"D:\data\AIVC\test_results\exp1b_replogle_hybrid_rpe1.csv", index=False, encoding="utf-8")
