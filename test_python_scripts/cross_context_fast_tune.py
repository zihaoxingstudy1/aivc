# -*- coding: utf-8 -*-
"""
跨细胞系低信噪比场景快速调优验证：
验证 Control Z-score 上下文编码 + Geneformer 基因先验 + 保守残差校准在 Replogle 与 sci-Plex3 上的表现。
"""

import os
import sys
import time
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


class CrossContextResNet(nn.Module):
    def __init__(self, n_genes: int, n_tokens: int, prior_mat: np.ndarray, hidden_dim: int = 128, use_dose: bool = False):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior_mat, dtype=torch.float32))
        self.tok_emb = nn.Embedding(n_tokens + 1, 32, padding_idx=0)
        self.prior_proj = nn.Linear(prior_mat.shape[1], hidden_dim - 32)
        self.ctx_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        in_dim = hidden_dim * 2 + (32 if use_dose else 0)
        self.use_dose = use_dose
        if use_dose:
            self.dose_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x_ctrl_mean, token_ids, delta_lin, dose_vals=None):
        x_z = (x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(x_z)
        mask = (token_ids == 0)
        valid = (~mask).unsqueeze(-1).float()
        h_tok = torch.cat([self.tok_emb(token_ids), self.prior_proj(self.prior[token_ids])], dim=-1)
        z_pert = (h_tok * valid).sum(dim=1)
        feats = [z_ctx, z_pert]
        if self.use_dose and dose_vals is not None:
            feats.append(self.dose_mlp(dose_vals.unsqueeze(-1)))
        res = self.net(torch.cat(feats, dim=-1))
        return delta_lin + torch.tanh(self.res_scale) * res


def run_fast_cross_context(h5ad_path, pert_col, sep_char, dose_col, ensembl_col, use_gf_gene_emb, model_name, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(h5ad_path)
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    def get_ctrl(sub):
        m = (sub.obs[pert_col].astype(str) == "control").values
        return _to_dense_array(sub.X[m]).mean(axis=0)

    mu_tr = get_ctrl(tr)
    mu_ev = get_ctrl(ev)
    tr_d = _compute_group_deltas(tr[tr.obs[pert_col] != "control"], pert_col, mu_tr)
    ev_d = _compute_group_deltas(ev[ev.obs[pert_col] != "control"], pert_col, mu_ev)

    def parse_toks(ps):
        bp = ps.split("__dose_")[0] if "__dose_" in ps else ps
        return [t.strip() for t in bp.split(sep_char) if t.strip()]

    all_p = sorted(set(tr_d.keys()) | set(ev_d.keys()))
    vocab = sorted({t for p in all_p for t in parse_toks(p)})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = max(len(parse_toks(p)) for p in all_p)

    if use_gf_gene_emb:
        gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
        tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
        gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
        var_ens = {g: str(adata.var[ensembl_col].iloc[i]) for i, g in enumerate(adata.var_names)} if ensembl_col in adata.var.columns else None
        embs = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
        p_dim = len(next(iter(embs.values()))) if embs else 64
        p_mat = np.zeros((len(vocab) + 1, p_dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in embs:
                p_mat[tid] = embs[t]
    else:
        p_mat = np.random.normal(0, 0.05, size=(len(vocab) + 1, 64)).astype(np.float32)

    tr_vocab = sorted({t for p in tr_d for t in parse_toks(p)})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(plist):
            for t in parse_toks(ps):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    tr_keys = sorted(tr_d.keys())
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_lin(tr_keys), np.vstack([tr_d[p] for p in tr_keys]))

    def make_t(d_dict, mu_c):
        ks = sorted(d_dict.keys())
        tids = np.zeros((len(ks), max_len), dtype=np.int64)
        doses = np.zeros(len(ks), dtype=np.float32)
        for i, p in enumerate(ks):
            for j, t in enumerate(parse_toks(p)[:max_len]):
                tids[i, j] = t2id.get(t, 0)
            if "__dose_" in p:
                doses[i] = float(np.log10(float(p.split("__dose_")[1]) + 1.0))
        dl = ridge.predict(enc_lin(ks)).astype(np.float32)
        yt = np.vstack([d_dict[p] for p in ks]).astype(np.float32)
        xc = np.tile(mu_c, (len(ks), 1)).astype(np.float32)
        return (ks, torch.tensor(xc, device=device), torch.tensor(tids, device=device), torch.tensor(dl, device=device), torch.tensor(yt, device=device), torch.tensor(doses, device=device))

    tr_k, tr_xc, tr_tid, tr_dl, tr_y, tr_do = make_t(tr_d, mu_tr)
    ev_k, ev_xc, ev_tid, ev_dl, ev_y, ev_do = make_t(ev_d, mu_ev)

    net = CrossContextResNet(adata.n_vars, len(vocab), p_mat, hidden_dim=128, use_dose=bool(dose_col)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(1, 61):
        net.train()
        opt.zero_grad()
        pred = net(tr_xc, tr_tid, tr_dl, tr_do if dose_col else None)
        loss = F.mse_loss(pred, tr_y) * 50.0 + (1.0 - F.cosine_similarity(pred, tr_y, dim=-1)).mean() * 0.5
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        epred = net(ev_xc, ev_tid, ev_dl, ev_do if dose_col else None).cpu().numpy()
    df_p = evaluate_perturbation_predictions(ev_d, {p: epred[i] for i, p in enumerate(ev_k)}, model_name, (20, 50))
    print(f"[{model_name}] Pearson All={df_p['pearson_delta_all'].mean():.4f} | Top20={df_p['pearson_delta_top20'].mean():.4f} | MSE={df_p['mse_all'].mean():.6f}")
    df_p.to_csv(out_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    run_fast_cross_context(
        r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad",
        "perturbation", "_", "", "ensembl_id", True,
        "DistillCell-GF316M-CrossCtx", r"D:\data\AIVC\test_results\replogle_gf316_crossctx_per_pert.csv"
    )
    run_fast_cross_context(
        r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad",
        "pert_dose", "||", "log10_dose", "ensembl_id", False,
        "BioResCell-Drug-CrossCtx", r"D:\data\AIVC\test_results\sciplex3_biores_crossctx_per_pert.csv"
    )
