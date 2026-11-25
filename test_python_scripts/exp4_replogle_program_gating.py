# -*- coding: utf-8 -*-
"""
弱项 1 破局实验：基因程序低秩空间 (Gene Program SVD Space) + 靶基因基态表达门控 + 多教师融合
针对 Replogle (K562 -> RPE1) 跨细胞系泛化：
1. 将高维噪声表达偏移压缩到 64 维跨细胞系保守基因程序空间 (SVD Basis)，参数量减少 50 倍，彻底抑制过拟合；
2. 引入靶基因自身在源细胞系/目标细胞系的基态表达特征 (Target Baseline Expression Gating)；
3. 融合 Geneformer-316M + scGPT + scFoundation 三教师语义，重点突破 340 个 RPE1 独有未见基因！
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_scfoundation_gene_embeddings


class GeneProgramCrossLineageNet(nn.Module):
    """
    低秩基因程序空间的跨细胞系预测网络：
    输出 64 维基因程序系数残差 + 全局标量缩放门控，绝不在 3600 维上直接过拟合。
    """
    def __init__(self, teacher_dims: list, program_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 48), nn.LayerNorm(48), nn.SiLU())
            for d in teacher_dims
        ])
        # 靶基因基态表达特征 (3维: ctrl_expr_target_cl, ctrl_expr_diff, is_expressed) -> 16维
        self.expr_gate_enc = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
        )
        in_dim = 48 * len(teacher_dims) + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.prog_head = nn.Linear(hidden_dim // 2, program_dim)
        nn.init.zeros_(self.prog_head.weight)
        nn.init.zeros_(self.prog_head.bias)
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, t_feats: list, target_expr_feat: torch.Tensor, basis_V: torch.Tensor, delta_lin: torch.Tensor):
        hs = [proj(f) for proj, f in zip(self.projs, t_feats)]
        he = self.expr_gate_enc(target_expr_feat)
        h = self.mlp(torch.cat(hs + [he], dim=-1))
        prog_coef = self.prog_head(h)  # [B, 64]
        delta_res = prog_coef @ basis_V  # [B, G]
        return delta_lin + torch.tanh(self.res_scale) * delta_res


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad")
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    tr_cm = (tr.obs["perturbation"].astype(str) == "control").values
    ev_cm = (ev.obs["perturbation"].astype(str) == "control").values
    mu_k562 = _to_dense_array(tr.X[tr_cm]).mean(axis=0)
    mu_rpe1 = _to_dense_array(ev.X[ev_cm]).mean(axis=0)

    tr_deltas = _compute_group_deltas(tr[~tr_cm], "perturbation", mu_k562)
    ev_deltas = _compute_group_deltas(ev[~ev_cm], "perturbation", mu_rpe1)

    tr_keys = sorted(tr_deltas.keys())
    ev_keys = sorted(ev_deltas.keys())
    shared_set = set(tr_keys) & set(ev_keys)
    unseen_set = set(ev_keys) - set(tr_keys)

    # 1. 构建 64 维跨细胞系保守基因程序基底 (TruncatedSVD on Train Deltas)
    Y_tr = np.vstack([tr_deltas[p] for p in tr_keys]).astype(np.float32)
    svd = TruncatedSVD(n_components=64, random_state=42)
    svd.fit(Y_tr)
    basis_V = torch.tensor(svd.components_.astype(np.float32), device=device)  # [64, G]
    print(f"[Exp4] 64 维基因程序解释方差比例: {svd.explained_variance_ratio_.sum():.4f}")

    # 2. 提取三教师嵌入
    vocab = sorted(set(tr_keys) | set(ev_keys))
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    var_ens = {g: str(adata.var["ensembl_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensembl_id" in adata.var.columns else None

    gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)

    gene_to_col = {g: i for i, g in enumerate(adata.var_names)}

    def get_teacher_and_expr_feats(plist, mu_ctrl_ctx):
        t_list = []
        for d in [gf_dict, scgpt_dict, scf_dict]:
            dim = len(next(iter(d.values())))
            mat = np.vstack([d.get(p, np.zeros(dim, dtype=np.float32)) for p in plist]).astype(np.float32)
            t_list.append(torch.tensor(mat, device=device))

        expr_feats = np.zeros((len(plist), 3), dtype=np.float32)
        for i, p in enumerate(plist):
            if p in gene_to_col:
                val = float(mu_ctrl_ctx[gene_to_col[p]])
                val_src = float(mu_k562[gene_to_col[p]])
                expr_feats[i, 0] = val
                expr_feats[i, 1] = val - val_src
                expr_feats[i, 2] = 1.0 if val > 0.1 else 0.0
        return t_list, torch.tensor(expr_feats, device=device)

    # 3. 强线性基准 (Ridge + 温和正则)
    tr_idx = {p: i for i, p in enumerate(tr_keys)}
    def enc_hot(plist):
        m = np.zeros((len(plist), len(tr_keys)), dtype=np.float32)
        for i, p in enumerate(plist):
            if p in tr_idx:
                m[i, tr_idx[p]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_hot(tr_keys), Y_tr)
    pred_tr_lin = ridge.predict(enc_hot(tr_keys)).astype(np.float32)
    pred_ev_lin = ridge.predict(enc_hot(ev_keys)).astype(np.float32)

    # 4. 利用 K562 内部 85/15 划分训练与验证（绝对不接触 RPE1 测试集！）
    rng = np.random.default_rng(42)
    perm_tr = rng.permutation(len(tr_keys))
    n_val = int(0.15 * len(tr_keys))
    val_ids = perm_tr[:n_val]
    fit_ids = perm_tr[n_val:]

    # 为了让模型真正学会预测未见基因（simulate unseen genes），fit 集上的 delta_lin 故意置一部分为 train_mean！
    global_mean = Y_tr[fit_ids].mean(axis=0, keepdims=True)
    t_feats_tr, expr_tr = get_teacher_and_expr_feats(tr_keys, mu_k562)
    t_feats_ev, expr_ev = get_teacher_and_expr_feats(ev_keys, mu_rpe1)

    t_y_tr = torch.tensor(Y_tr, device=device)
    t_dl_tr = torch.tensor(pred_tr_lin, device=device)
    t_dl_ev = torch.tensor(pred_ev_lin, device=device)
    t_gmean = torch.tensor(global_mean, device=device)

    teacher_dims = [f.shape[1] for f in t_feats_tr]
    net = GeneProgramCrossLineageNet(teacher_dims, program_dim=64, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=2e-2)

    best_val_loss = 1e9
    best_st = None
    for ep in range(1, 121):
        net.train()
        opt.zero_grad()
        f_sub = [f[fit_ids] for f in t_feats_tr]
        e_sub = expr_tr[fit_ids]
        # 关键训练技巧：30% 概率用 global_mean 替换 delta_lin，强迫网络学会仅靠三教师语义预测“未见基因”！
        dl_sub = t_dl_tr[fit_ids].clone()
        drop_mask = (torch.rand(len(fit_ids), 1, device=device) < 0.30)
        dl_sub = torch.where(drop_mask, t_gmean.expand_as(dl_sub), dl_sub)

        pred = net(f_sub, e_sub, basis_V, dl_sub)
        y_sub = t_y_tr[fit_ids]
        loss = F.mse_loss(pred, y_sub) * 50.0 + (1.0 - F.cosine_similarity(pred, y_sub, dim=-1)).mean() * 0.5
        loss.backward()
        opt.step()

        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                v_f = [f[val_ids] for f in t_feats_tr]
                # 验证时同时测已知基因和模拟未见基因
                v_pred = net(v_f, expr_tr[val_ids], basis_V, t_gmean.expand(len(val_ids), -1))
                v_loss = float(F.mse_loss(v_pred, t_y_tr[val_ids]).item() - 0.01 * F.cosine_similarity(v_pred, t_y_tr[val_ids], dim=-1).mean().item())
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_st is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_st.items()})

    net.eval()
    with torch.no_grad():
        pred_ev_gp = net(t_feats_ev, expr_ev, basis_V, t_dl_ev).cpu().numpy()

    df_ridge = evaluate_perturbation_predictions(ev_deltas, {p: pred_ev_lin[i] for i, p in enumerate(ev_keys)}, "Linear-Ridge(K562->RPE1)", (20, 50))
    df_gp = evaluate_perturbation_predictions(ev_deltas, {p: pred_ev_gp[i] for i, p in enumerate(ev_keys)}, "GeneProgram-TriTeacher", (20, 50))

    for name, df in [("Linear-Ridge", df_ridge), ("GeneProgram-TriTeacher", df_gp)]:
        sh = df[df["perturbation"].isin(shared_set)]
        un = df[df["perturbation"].isin(unseen_set)]
        print(f"\n=== [{name}] RPE1 测试集表现 ===")
        print(f"  全部 2016 扰动 : Pearson={df['pearson_delta_all'].mean():.4f} | Top20={df['pearson_delta_top20'].mean():.4f} | MSE={df['mse_all'].mean():.6f}")
        print(f"  共有 1676 扰动 : Pearson={sh['pearson_delta_all'].mean():.4f} | Top20={sh['pearson_delta_top20'].mean():.4f} | MSE={sh['mse_all'].mean():.6f}")
        print(f"  未见  340 扰动 : Pearson={un['pearson_delta_all'].mean():.4f} | Top20={un['pearson_delta_top20'].mean():.4f} | MSE={un['mse_all'].mean():.6f}")

    pd.concat([df_ridge, df_gp], ignore_index=True).to_csv(r"D:\data\AIVC\test_results\exp4_replogle_geneprogram_rpe1.csv", index=False, encoding="utf-8")
