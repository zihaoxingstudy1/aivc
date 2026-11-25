# -*- coding: utf-8 -*-
"""
AIVC-OmniFusion: 门控低秩多教师融合 + 细胞上下文对齐 + 残差动力学融合架构
解决多教师直接拼接维度灾难，支持任意数量教师大模型与任意生物场景的动态权重融合。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


class GatedMultiTeacherFusionNet(nn.Module):
    """
    门控低秩多教师融合架构:
    1. 每个教师嵌入独立投影到统一低秩空间 (bottleneck_dim=64) 并做 LayerNorm；
    2. 细胞基态语境 z_ctx 生成 Softmax 门控权重，动态加权融合各教师表征；
    3. Set-Attention + 二阶上位交互捕获组合协同；
    4. 结合线性先验门控与残差头输出最终扰动偏移（并支持可选细胞上下文表征蒸馏）。
    """
    def __init__(
        self,
        n_genes: int,
        n_tokens: int,
        teacher_matrices: list,
        bottleneck_dim: int = 64,
        hidden_dim: int = 192,
        teacher_ctx_dim: int = 0,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_teachers = len(teacher_matrices)
        self.bottleneck_dim = bottleneck_dim

        # 注册各教师冻结嵌入矩阵
        for idx, mat in enumerate(teacher_matrices):
            self.register_buffer(f"t_mat_{idx}", torch.tensor(mat, dtype=torch.float32))

        self.tok_emb = nn.Embedding(n_tokens + 1, bottleneck_dim, padding_idx=0)
        self.teacher_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mat.shape[1], bottleneck_dim),
                    nn.LayerNorm(bottleneck_dim),
                    nn.SiLU(),
                )
                for mat in teacher_matrices
            ]
        )

        # 细胞基态语境编码器 (Z-score 归一化输入)
        self.ctx_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 上下文感知的多教师动态门控权重 (输出 n_teachers + 1 维: 各教师 + 可学习token嵌入)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.n_teachers + 1),
        )

        # 扰动集合编码与二阶交互
        self.pert_enc = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.set_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.use_ctx_distill = teacher_ctx_dim > 0
        if self.use_ctx_distill:
            self.ctx_distill_head = nn.Linear(hidden_dim, teacher_ctx_dim)

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.prior_gate_raw = nn.Parameter(torch.zeros(1, n_genes))

    def forward(self, x_ctrl_mean: torch.Tensor, token_ids: torch.Tensor, delta_lin: torch.Tensor):
        # 1. Z-score 语境编码
        xz = (x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(xz)  # [B, H]

        # 2. 动态门控权重 [B, K+1]
        gate_weights = F.softmax(self.gate_mlp(z_ctx), dim=-1)  # [B, K+1]

        # 3. 投影所有教师特征到统一 64d 低秩空间并按门控加权融合
        e_list = [self.tok_emb(token_ids)]  # [B, L, 64]
        for idx, proj in enumerate(self.teacher_projs):
            t_buf = getattr(self, f"t_mat_{idx}")
            e_list.append(proj(t_buf[token_ids]))  # [B, L, 64]

        stacked = torch.stack(e_list, dim=2)  # [B, L, K+1, 64]
        fused_tok = (stacked * gate_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # [B, L, 64]

        # 4. Set-Attention + 二阶上位交互
        mask = (token_ids == 0)
        valid = (~mask).unsqueeze(-1).float()
        h_tok = self.pert_enc(fused_tok)  # [B, L, H]
        attn_out, _ = self.set_attn(h_tok, h_tok, h_tok, key_padding_mask=mask)
        h_tok = h_tok + attn_out
        h_sum = (h_tok * valid).sum(dim=1)
        if token_ids.shape[1] >= 2:
            h_pair = (h_tok[:, 0, :] * h_tok[:, 1, :]) * (valid[:, 0, :] * valid[:, 1, :])
        else:
            h_pair = torch.zeros_like(h_sum)

        feat = self.fusion(torch.cat([z_ctx, h_sum, h_pair], dim=-1))
        prior_gate = 1.0 + 0.25 * torch.tanh(self.prior_gate_raw)
        delta_pred = delta_lin * prior_gate + self.delta_head(feat)

        if self.use_ctx_distill:
            t_ctx_pred = F.normalize(self.ctx_distill_head(feat), dim=-1)
            return delta_pred, t_ctx_pred, gate_weights
        return delta_pred, None, gate_weights


def train_and_eval_omni_fusion(
    adata_or_path,
    teacher_gene_dicts: dict,
    teacher_cell_embs: dict = None,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    split_col: str = "ood_split",
    ridge_alpha: float = 10.0,
    bottleneck_dim: int = 64,
    hidden_dim: int = 192,
    dropout: float = 0.25,
    epochs: int = 100,
    lr: float = 6e-4,
    weight_decay: float = 5e-3,
    lambda_distill: float = 0.3,
    lambda_mse: float = 25.0,
    top_k_str: str = "20;50",
    seed: int = 42,
    model_name: str = "AIVC-OmniFusion",
    save_ckpt_path: str = "",
    per_pert_csv_path: str = "",
    summary_csv_path: str = "",
):
    """训练并评测门控低秩多教师融合模型 AIVC-OmniFusion。"""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k_list = tuple(int(x.strip()) for x in str(top_k_str).split(";") if x.strip())

    adata = sc.read_h5ad(adata_or_path) if isinstance(adata_or_path, str) else adata_or_path
    tr = adata[adata.obs[split_col] == "train"]
    va = adata[adata.obs[split_col] == "val"] if (adata.obs[split_col] == "val").sum() > 0 else None
    ev = adata[adata.obs[split_col] == "test"]

    def get_cm(sub):
        m = (sub.obs[pert_col].astype(str) == control_label).values
        return _to_dense_array(sub.X[m]).mean(axis=0)

    mu_tr, mu_ev = get_cm(tr), get_cm(ev)
    mu_va = get_cm(va) if va is not None else None
    tr_d = _compute_group_deltas(tr[tr.obs[pert_col] != control_label], pert_col, mu_tr)
    ev_d = _compute_group_deltas(ev[ev.obs[pert_col] != control_label], pert_col, mu_ev)
    va_d = _compute_group_deltas(va[va.obs[pert_col] != control_label], pert_col, mu_va) if va is not None else None

    def parse_toks(ps):
        bp = ps.split("__dose_")[0] if "__dose_" in ps else ps
        return [t.strip() for t in bp.split(sep_char) if t.strip()]

    all_p = sorted(set(tr_d) | set(ev_d) | (set(va_d) if va_d else set()))
    vocab = sorted({t for p in all_p for t in parse_toks(p)})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = max(len(parse_toks(p)) for p in all_p)

    # 构建每个教师的独立矩阵
    teacher_names = list(teacher_gene_dicts.keys())
    teacher_matrices = []
    for tn in teacher_names:
        td = teacher_gene_dicts[tn]
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

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
    ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    ridge.fit(enc_lin(tr_keys), np.vstack([tr_d[p] for p in tr_keys]))

    t_ctx_dim = len(next(iter(teacher_cell_embs.values()))) if (teacher_cell_embs and len(teacher_cell_embs) > 0) else 0

    def make_t(d_dict, mu_c):
        ks = sorted(d_dict.keys())
        tids = np.zeros((len(ks), max_len), dtype=np.int64)
        for i, p in enumerate(ks):
            for j, t in enumerate(parse_toks(p)[:max_len]):
                tids[i, j] = t2id.get(t, 0)
        dl = ridge.predict(enc_lin(ks)).astype(np.float32)
        yt = np.vstack([d_dict[p] for p in ks]).astype(np.float32)
        xc = np.tile(mu_c, (len(ks), 1)).astype(np.float32)
        if t_ctx_dim > 0:
            tc = np.vstack([teacher_cell_embs.get(p, np.zeros(t_ctx_dim, dtype=np.float32)) for p in ks])
            tc_t = torch.tensor(tc, device=device, dtype=torch.float32)
        else:
            tc_t = None
        return (ks, torch.tensor(xc, device=device), torch.tensor(tids, device=device), torch.tensor(dl, device=device), torch.tensor(yt, device=device), tc_t)

    tr_k, tr_xc, tr_tid, tr_dl, tr_y, tr_tc = make_t(tr_d, mu_tr)
    ev_k, ev_xc, ev_tid, ev_dl, ev_y, _ = make_t(ev_d, mu_ev)
    if va_d is not None:
        va_k, va_xc, va_tid, va_dl, va_y, _ = make_t(va_d, mu_va)

    net = GatedMultiTeacherFusionNet(
        n_genes=adata.n_vars,
        n_tokens=len(vocab),
        teacher_matrices=teacher_matrices,
        bottleneck_dim=int(bottleneck_dim),
        hidden_dim=int(hidden_dim),
        teacher_ctx_dim=t_ctx_dim,
        dropout=float(dropout),
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(lr) * 0.05)

    best_score = -1e9
    best_st = None
    for ep in range(1, int(epochs) + 1):
        net.train()
        opt.zero_grad()
        pred_d, pred_tc, _ = net(tr_xc, tr_tid, tr_dl)
        loss = (
            F.smooth_l1_loss(pred_d, tr_y, beta=0.1)
            + float(lambda_mse) * F.mse_loss(pred_d, tr_y)
            + 0.8 * (1.0 - F.cosine_similarity(pred_d, tr_y, dim=-1)).mean()
        )
        if t_ctx_dim > 0 and pred_tc is not None and tr_tc is not None:
            loss = loss + float(lambda_distill) * (1.0 - F.cosine_similarity(pred_tc, tr_tc, dim=-1)).mean()
        loss.backward()
        opt.step()
        sched.step()

        if (ep % 10 == 0 or ep == int(epochs)) and va_d is not None:
            net.eval()
            with torch.no_grad():
                vp, _, _ = net(va_xc, va_tid, va_dl)
                vp_np = vp.cpu().numpy()
                dfv = evaluate_perturbation_predictions(va_d, {p: vp_np[i] for i, p in enumerate(va_k)}, model_name, top_k_list)
                sc_v = float(dfv["pearson_delta_all"].mean() + 0.5 * dfv[f"pearson_delta_top{top_k_list[0]}"].mean() - 10.0 * dfv["mse_all"].mean())
                if sc_v > best_score:
                    best_score = sc_v
                    best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_st is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_st.items()})

    if save_ckpt_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_ckpt_path)), exist_ok=True)
        torch.save(net.state_dict(), save_ckpt_path)

    net.eval()
    with torch.no_grad():
        ep_pred, _, gw = net(ev_xc, ev_tid, ev_dl)
        ep_np = ep_pred.cpu().numpy()
        mean_gw = gw.mean(dim=0).cpu().numpy()

    gate_labels = ["TokenEmb"] + teacher_names
    gw_str = " | ".join(f"{lbl}:{w:.3f}" for lbl, w in zip(gate_labels, mean_gw))
    print(f"[{model_name}] 动态教师门控权重 -> {gw_str}")

    per_pert_df = evaluate_perturbation_predictions(ev_d, {p: ep_np[i] for i, p in enumerate(ev_k)}, model_name, top_k_list)
    metric_cols = [c for c in per_pert_df.columns if c not in ("model", "perturbation")]
    srow = {"model": model_name, "n_eval_perts": int(len(per_pert_df))}
    for mc in metric_cols:
        srow[f"{mc}_mean"] = float(per_pert_df[mc].mean())
        srow[f"{mc}_std"] = float(per_pert_df[mc].std())
    summary_df = pd.DataFrame([srow])

    print(
        f"[{model_name}] 测试集表现: Pearson All={srow['pearson_delta_all_mean']:.4f} | "
        f"Top{top_k_list[0]}={srow[f'pearson_delta_top{top_k_list[0]}_mean']:.4f} | "
        f"DirAcc={srow[f'dir_acc_top{top_k_list[0]}_mean']:.4f} | MSE={srow['mse_all_mean']:.6f}"
    )

    if per_pert_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(per_pert_csv_path)), exist_ok=True)
        per_pert_df.to_csv(per_pert_csv_path, index=False, encoding="utf-8")
    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    return summary_df, per_pert_df


# 中文注释：AIVC-OmniFusion 门控低秩多教师融合模型，融合多个单细胞与蛋白大模型并支持细胞上下文表征蒸馏。
