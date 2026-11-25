# -*- coding: utf-8 -*-
"""
方向 1 实验：多谱系联合训练 (K562 + Norman_K562 + Primary_T + THP1 -> RPE1)
使用 Per-Context Delta（按各自细胞系减去 Control 均值）+ Geneformer-V2-316M 基因先验，攻克 RPE1 跨细胞系泛化。
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


class MultiLineageCrossCtxNet(nn.Module):
    def __init__(self, n_genes: int, n_tokens: int, prior_mat: np.ndarray, hidden_dim: int = 192):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior_mat, dtype=torch.float32))
        self.tok_emb = nn.Embedding(n_tokens + 1, 64, padding_idx=0)
        self.prior_proj = nn.Linear(prior_mat.shape[1], hidden_dim - 64)
        self.pert_enc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x_ctrl_mean, token_ids, delta_lin):
        xz = (x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(xz)
        mask = (token_ids == 0)
        valid = (~mask).unsqueeze(-1).float()
        h_tok = self.pert_enc(torch.cat([self.tok_emb(token_ids), self.prior_proj(self.prior[token_ids])], dim=-1))
        z_pert = (h_tok * valid).sum(dim=1)
        res = self.fusion(torch.cat([z_ctx, z_pert], dim=-1))
        return delta_lin + torch.tanh(self.res_scale) * res


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(r"D:\data\AIVC\data\MultiLineage_Replogle_RPE1.h5ad")
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    # 1. Per-Context 计算训练集各 (cell_line, perturbation) 的真实 Delta
    tr_samples = []  # (cell_line, pert, mu_ctrl_ctx, delta_true)
    ctx_ctrl_means = {}
    for cl in sorted(tr.obs["cell_line"].unique()):
        sub_cl = tr[tr.obs["cell_line"] == cl]
        c_mask = (sub_cl.obs["perturbation"].astype(str) == "control").values
        if c_mask.sum() == 0:
            continue
        mu_c = _to_dense_array(sub_cl.X[c_mask]).mean(axis=0)
        ctx_ctrl_means[cl] = mu_c
        d_dict = _compute_group_deltas(sub_cl[~c_mask], "perturbation", mu_c)
        for p, dt in d_dict.items():
            tr_samples.append((cl, p, mu_c, dt))

    print(f"[Exp1] 多谱系训练样本组总数: {len(tr_samples)} (覆盖细胞系: {list(ctx_ctrl_means.keys())})")

    # 测试集 RPE1
    ev_c_mask = (ev.obs["perturbation"].astype(str) == "control").values
    mu_rpe1 = _to_dense_array(ev.X[ev_c_mask]).mean(axis=0)
    ev_deltas = _compute_group_deltas(ev[~ev_c_mask], "perturbation", mu_rpe1)
    ev_keys = sorted(ev_deltas.keys())

    def parse_toks(ps):
        return [t.strip() for t in ps.split("_") if t.strip()]

    all_p = sorted({s[1] for s in tr_samples} | set(ev_keys))
    vocab = sorted({t for p in all_p for t in parse_toks(p)})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = max(len(parse_toks(p)) for p in all_p)

    # 2. Per-Context 线性基线 (Linear-Ridge Per-Context)
    tr_vocab = sorted({t for s in tr_samples for t in parse_toks(s[1])})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(plist):
            for t in parse_toks(ps):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_lin([s[1] for s in tr_samples]), np.vstack([s[3] for s in tr_samples]))

    ev_ridge_pred = ridge.predict(enc_lin(ev_keys))
    df_ridge = evaluate_perturbation_predictions(ev_deltas, {p: ev_ridge_pred[i] for i, p in enumerate(ev_keys)}, "Linear-Ridge(MultiCtx-PerCtrl)", (20, 50))
    print(f"[Exp1] Linear-Ridge(MultiCtx-PerCtrl) -> RPE1: Pearson={df_ridge['pearson_delta_all'].mean():.4f} | Top20={df_ridge['pearson_delta_top20'].mean():.4f} | MSE={df_ridge['mse_all'].mean():.6f}")

    # 3. 提取 Geneformer-V2-316M 1152维先验
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    var_ens = {g: str(adata.var["ensembl_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensembl_id" in adata.var.columns else None
    embs = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    p_dim = len(next(iter(embs.values())))
    p_mat = np.zeros((len(vocab) + 1, p_dim), dtype=np.float32)
    for t, tid in t2id.items():
        if t in embs:
            p_mat[tid] = embs[t]

    # 4. 构造训练与测试张量
    def make_tids(plist):
        arr = np.zeros((len(plist), max_len), dtype=np.int64)
        for i, p in enumerate(plist):
            for j, t in enumerate(parse_toks(p)[:max_len]):
                arr[i, j] = t2id.get(t, 0)
        return torch.tensor(arr, device=device)

    tr_xc = torch.tensor(np.vstack([s[2] for s in tr_samples]), device=device, dtype=torch.float32)
    tr_tid = make_tids([s[1] for s in tr_samples])
    tr_dl = torch.tensor(ridge.predict(enc_lin([s[1] for s in tr_samples])), device=device, dtype=torch.float32)
    tr_y = torch.tensor(np.vstack([s[3] for s in tr_samples]), device=device, dtype=torch.float32)

    ev_xc = torch.tensor(np.tile(mu_rpe1, (len(ev_keys), 1)), device=device, dtype=torch.float32)
    ev_tid = make_tids(ev_keys)
    ev_dl = torch.tensor(ev_ridge_pred, device=device, dtype=torch.float32)

    net = MultiLineageCrossCtxNet(adata.n_vars, len(vocab), p_mat, hidden_dim=192).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(1, 81):
        net.train()
        opt.zero_grad()
        pred = net(tr_xc, tr_tid, tr_dl)
        loss = F.mse_loss(pred, tr_y) * 50.0 + (1.0 - F.cosine_similarity(pred, tr_y, dim=-1)).mean() * 0.5
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        epred = net(ev_xc, ev_tid, ev_dl).cpu().numpy()
    df_ml = evaluate_perturbation_predictions(ev_deltas, {p: epred[i] for i, p in enumerate(ev_keys)}, "MultiLineage-GF316M", (20, 50))
    print(f"[Exp1] MultiLineage-GF316M -> RPE1: Pearson={df_ml['pearson_delta_all'].mean():.4f} | Top20={df_ml['pearson_delta_top20'].mean():.4f} | MSE={df_ml['mse_all'].mean():.6f}")

    out_df = pd.concat([df_ridge, df_ml], ignore_index=True)
    out_df.to_csv(r"D:\data\AIVC\test_results\exp1_multilineage_rpe1_per_pert.csv", index=False, encoding="utf-8")
