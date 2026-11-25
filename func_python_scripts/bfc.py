# -*- coding: utf-8 -*-
"""
BioFlowCell: 生物先验残差条件流匹配虚拟细胞模型 (原创核心模型 + 蒸馏兼容接口)
融合线性先验残差主干、基因共表达/蛋白先验、Set-Attention 二阶交互、细胞基态语境编码与 OT-CFM 分布迁移。
"""

import os
import math
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


def build_gene_coexp_prior(X_train_ctrl_and_pert: np.ndarray, gene_names: list, prior_dim: int = 64) -> dict:
    """
    基于训练集表达矩阵构建基因共表达 SVD 先验字典（零外部依赖，纯生物表达先验）。
    """
    # 转置为 [n_genes, n_cells_sample]
    n_sample = min(15000, X_train_ctrl_and_pert.shape[0])
    X_sub = X_train_ctrl_and_pert[:n_sample]
    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()
    X_g = X_sub.T.astype(np.float32)
    # 标准化每个基因
    means = X_g.mean(axis=1, keepdims=True)
    stds = X_g.std(axis=1, keepdims=True) + 1e-6
    X_g = (X_g - means) / stds
    svd = TruncatedSVD(n_components=min(prior_dim, X_g.shape[1] - 1, X_g.shape[0] - 1), random_state=42)
    gene_vecs = svd.fit_transform(X_g).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(gene_vecs, axis=1, keepdims=True) + 1e-6
    gene_vecs = gene_vecs / norms
    return {g: gene_vecs[i] for i, g in enumerate(gene_names)}


class SinusoidalTimeEmb(nn.Module):
    """正弦时间步编码。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResBlock(nn.Module):
    """条件 FiLM 调制残差块。"""
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(cond_dim, hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = self.norm1(h)
        x = x * (1.0 + gamma) + beta
        x = F.silu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(F.silu(self.norm2(x)))
        return h + self.drop(x)


class BioFlowCellNet(nn.Module):
    """
    BioFlowCell 主网络：
    - 细胞基态语境编码器 (Context Encoder)
    - 生物先验扰动集合编码器 (Set-Attention + 二阶上位交互 + 连续剂量编码)
    - 隐空间残差流匹配速度场 (Latent/Gene-Space Residual Velocity Field)
    """
    def __init__(
        self,
        n_genes: int,
        n_tokens: int,
        prior_dim: int = 64,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.15,
        use_dose: bool = False,
        prior_matrix: np.ndarray = None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.use_dose = use_dose

        # 1. 基因/药物 Token 嵌入 + 冻结生物先验融合
        self.token_emb = nn.Embedding(n_tokens + 1, hidden_dim // 2, padding_idx=0)
        if prior_matrix is not None:
            self.register_buffer("prior_mat", torch.tensor(prior_matrix, dtype=torch.float32))
            self.prior_proj = nn.Linear(prior_matrix.shape[1], hidden_dim // 2)
            pert_in_dim = hidden_dim
        else:
            self.prior_mat = None
            pert_in_dim = hidden_dim // 2

        self.pert_phi = nn.Sequential(
            nn.Linear(pert_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 集合自注意力与二阶交互融合
        self.set_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pert_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        if use_dose:
            self.dose_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )

        # 2. 细胞基态语境编码器 (编码当前 Control 细胞的状态)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 3. 速度场主干
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_proj = nn.Linear(n_genes, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([FiLMResBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_blocks)])
        self.vel_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        # 初始化最后一层接近 0，使模型起点平滑继承线性先验
        nn.init.zeros_(self.vel_head[-1].weight)
        nn.init.zeros_(self.vel_head[-1].bias)

        # 可学习的线性先验有界缩放门控（稳固线性主干，防止过拟合跑偏）
        self.prior_gate_raw = nn.Parameter(torch.zeros(1, n_genes))

    def get_prior_gate(self) -> torch.Tensor:
        return 1.0 + 0.25 * torch.tanh(self.prior_gate_raw)

    def encode_pert(self, token_ids: torch.Tensor, dose_vals: torch.Tensor = None) -> torch.Tensor:
        # token_ids: [B, max_len] (0 为 padding)
        mask = (token_ids == 0)  # True 表示 padding
        e = self.token_emb(token_ids)
        if self.prior_mat is not None:
            p_vec = self.prior_mat[token_ids]
            e = torch.cat([e, self.prior_proj(p_vec)], dim=-1)
        h = self.pert_phi(e)  # [B, L, H]

        # Set Self-Attention
        attn_out, _ = self.set_attn(h, h, h, key_padding_mask=mask)
        h = h + attn_out
        valid_mask = (~mask).unsqueeze(-1).float()
        h_sum = (h * valid_mask).sum(dim=1)
        # 二阶交互项 (捕获双基因上位性协同; 单基因时交互为 0)
        if token_ids.shape[1] >= 2:
            pair_prod = (h[:, 0, :] * h[:, 1, :]) * (valid_mask[:, 0, :] * valid_mask[:, 1, :])
        else:
            pair_prod = torch.zeros_like(h_sum)

        z_pert = self.pert_out(torch.cat([h_sum, pair_prod], dim=-1))
        if self.use_dose and dose_vals is not None:
            z_pert = z_pert + self.dose_mlp(dose_vals.unsqueeze(-1))
        return z_pert

    def forward(
        self,
        x_t: torch.Tensor,
        x_0_ctx: torch.Tensor,
        t: torch.Tensor,
        token_ids: torch.Tensor,
        dose_vals: torch.Tensor = None,
    ) -> torch.Tensor:
        z_ctx = self.ctx_encoder(x_0_ctx)
        z_pert = self.encode_pert(token_ids, dose_vals)
        z_t = self.time_mlp(self.time_emb(t))
        cond = self.cond_proj(torch.cat([z_ctx, z_pert, z_t], dim=-1))

        h = self.state_proj(x_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.vel_head(h)

    @torch.no_grad()
    def sample_ode(
        self,
        x_0: torch.Tensor,
        token_ids: torch.Tensor,
        delta_lin_prior: torch.Tensor,
        dose_vals: torch.Tensor = None,
        steps: int = 8,
    ) -> torch.Tensor:
        """
        使用 Midpoint ODE 积分从真实 Control 细胞 x_0 推演到扰动后细胞状态 x_1。
        """
        dt = 1.0 / float(steps)
        x = x_0.clone()
        bsz = x.shape[0]
        for s in range(steps):
            t_cur = torch.full((bsz,), s * dt, device=x.device, dtype=x.dtype)
            k1 = self.forward(x, x_0, t_cur, token_ids, dose_vals)
            t_mid = torch.full((bsz,), (s + 0.5) * dt, device=x.device, dtype=x.dtype)
            x_mid = x + 0.5 * dt * k1
            k2 = self.forward(x_mid, x_0, t_mid, token_ids, dose_vals)
            x = x + dt * k2
        # 叠加可调节的线性残差先验
        return x + delta_lin_prior * self.get_prior_gate()


class GroupPairedDataset(Dataset):
    """按扰动组组织的单细胞配对数据集（支持高效伪批量损失与流匹配）。"""
    def __init__(self, X_pert_list, token_ids_arr, dose_arr, delta_lin_arr):
        self.X_pert_list = X_pert_list
        self.token_ids_arr = token_ids_arr
        self.dose_arr = dose_arr
        self.delta_lin_arr = delta_lin_arr

    def __len__(self):
        return len(self.X_pert_list)

    def __getitem__(self, idx):
        return idx


def train_and_eval_bioflowcell(
    adata_or_path,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    split_col: str = "ood_split",
    context_col: str = "",
    dose_col: str = "",
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    prior_dim: int = 64,
    hidden_dim: int = 256,
    n_blocks: int = 3,
    dropout: float = 0.15,
    ridge_alpha: float = 10.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    cells_per_group_batch: int = 24,
    groups_per_step: int = 8,
    lambda_cfm: float = 0.25,
    lambda_cos: float = 1.0,
    lambda_huber: float = 1.0,
    lambda_mse: float = 20.0,
    ode_steps: int = 8,
    seed: int = 42,
    center_by_context: bool = True,
    teacher_gene_emb_dict: dict = None,
    model_name: str = "BioFlowCell",
    top_k_str: str = "20;50",
    save_ckpt_path: str = "",
    per_pert_csv_path: str = "",
    summary_csv_path: str = "",
):
    """
    训练并评测 BioFlowCell 模型（支持基因组合、跨细胞系与药物+剂量多场景）。
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] 使用设备: {device}")

    pert_col = str(pert_col)
    control_label = str(control_label)
    sep_char = str(sep_char)
    split_col = str(split_col)
    context_col = str(context_col) if context_col else ""
    dose_col = str(dose_col) if dose_col else ""
    top_k_list = tuple(int(x.strip()) for x in str(top_k_str).split(";") if x.strip())

    if isinstance(adata_or_path, str):
        adata = sc.read_h5ad(adata_or_path)
    else:
        adata = adata_or_path

    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    train_adata = adata[adata.obs[split_col] == train_split]
    val_adata = adata[adata.obs[split_col] == val_split] if (adata.obs[split_col] == val_split).sum() > 0 else None
    eval_adata = adata[adata.obs[split_col] == test_split]

    # 1. 提取各 split 的 Control 池（支持按 context 分桶匹配 Control！）
    def extract_ctrl_pool(sub_adata):
        c_mask = (sub_adata.obs[pert_col].astype(str) == control_label).values
        X_c = _to_dense_array(sub_adata.X[c_mask])
        if context_col and context_col in sub_adata.obs.columns:
            ctx_c = sub_adata.obs.loc[c_mask, context_col].astype(str).values
            pools = {ctx: X_c[ctx_c == ctx] for ctx in np.unique(ctx_c)}
            pools["__ALL__"] = X_c
            return pools
        return {"__ALL__": X_c}

    train_ctrl_pools = extract_ctrl_pool(train_adata)
    eval_ctrl_pools = extract_ctrl_pool(eval_adata)
    val_ctrl_pools = extract_ctrl_pool(val_adata) if val_adata is not None else None

    # 2. 构建扰动词表与先验矩阵
    train_non_ctrl = train_adata[train_adata.obs[pert_col].astype(str) != control_label]
    eval_non_ctrl = eval_adata[eval_adata.obs[pert_col].astype(str) != control_label]

    def parse_pert_tokens(p_str: str):
        # 兼容 drug__dose_1000 格式
        base_p = p_str.split("__dose_")[0] if "__dose_" in p_str else p_str
        return [t.strip() for t in base_p.split(sep_char) if t.strip()]

    all_non_ctrl_perts = sorted(
        set(train_non_ctrl.obs[pert_col].astype(str).unique())
        | set(eval_non_ctrl.obs[pert_col].astype(str).unique())
        | (set(val_adata.obs.loc[val_adata.obs[pert_col] != control_label, pert_col].astype(str).unique()) if val_adata is not None else set())
    )
    vocab_tokens = sorted({tok for p in all_non_ctrl_perts for tok in parse_pert_tokens(p)})
    token_to_id = {tok: i + 1 for i, tok in enumerate(vocab_tokens)}  # 0 保留给 padding
    max_len = max(len(parse_pert_tokens(p)) for p in all_non_ctrl_perts)

    # 构建基因共表达先验（或融合外部 Teacher 先验）
    coexp_dict = build_gene_coexp_prior(train_adata.X, gene_names, prior_dim=prior_dim)
    if teacher_gene_emb_dict is not None and len(teacher_gene_emb_dict) > 0:
        t_dim = len(next(iter(teacher_gene_emb_dict.values())))
        full_prior_dim = prior_dim + t_dim
        prior_matrix = np.zeros((len(vocab_tokens) + 1, full_prior_dim), dtype=np.float32)
        for tok, tid in token_to_id.items():
            v1 = coexp_dict.get(tok, np.zeros(prior_dim, dtype=np.float32))
            v2 = teacher_gene_emb_dict.get(tok, np.zeros(t_dim, dtype=np.float32))
            prior_matrix[tid] = np.concatenate([v1, v2], axis=0)
        print(f"[{model_name}] 已融合基因共表达({prior_dim}d) + 外部教师先验({t_dim}d)")
    else:
        prior_matrix = np.zeros((len(vocab_tokens) + 1, prior_dim), dtype=np.float32)
        for tok, tid in token_to_id.items():
            if tok in coexp_dict:
                prior_matrix[tid] = coexp_dict[tok]

    # 3. 拟合一阶线性残差先验 (Linear-Ridge on Train)
    mu_ctrl_train = train_ctrl_pools["__ALL__"].mean(axis=0)
    train_pert_deltas = _compute_group_deltas(train_non_ctrl, pert_col, mu_ctrl_train)
    train_vocab = sorted({tok for p in train_pert_deltas.keys() for tok in parse_pert_tokens(p)})
    lin_tok_idx = {t: i for i, t in enumerate(train_vocab)}

    def encode_lin(pert_list):
        m = np.zeros((len(pert_list), len(train_vocab)), dtype=np.float32)
        for r_i, p_str in enumerate(pert_list):
            for tok in parse_pert_tokens(p_str):
                if tok in lin_tok_idx:
                    m[r_i, lin_tok_idx[tok]] = 1.0
        return m

    train_p_keys = sorted(train_pert_deltas.keys())
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(encode_lin(train_p_keys), np.vstack([train_pert_deltas[p] for p in train_p_keys]))

    def get_group_tensors(sub_non_ctrl):
        perts = sub_non_ctrl.obs[pert_col].astype(str).values
        ctxs = sub_non_ctrl.obs[context_col].astype(str).values if (context_col and context_col in sub_non_ctrl.obs.columns) else np.array(["__ALL__"] * len(perts))
        doses = sub_non_ctrl.obs[dose_col].astype(np.float32).values if (dose_col and dose_col in sub_non_ctrl.obs.columns) else np.zeros(len(perts), dtype=np.float32)

        df_idx = pd.DataFrame({"p": perts, "c": ctxs, "d": doses, "row": np.arange(len(perts))})
        groups = []
        for p_name, grp in df_idx.groupby("p", sort=True):
            rows = grp["row"].values
            X_g = _to_dense_array(sub_non_ctrl.X[rows])
            ctx_g = grp["c"].iloc[0]
            dose_g = float(grp["d"].mean())
            toks = parse_pert_tokens(p_name)
            tid_arr = np.zeros(max_len, dtype=np.int64)
            for j, t in enumerate(toks[:max_len]):
                tid_arr[j] = token_to_id.get(t, 0)
            delta_lin = ridge.predict(encode_lin([p_name]))[0].astype(np.float32)
            groups.append(
                {
                    "pert": p_name,
                    "ctx": ctx_g,
                    "X": X_g,
                    "token_ids": tid_arr,
                    "dose": dose_g,
                    "delta_lin": delta_lin,
                }
            )
        return groups

    train_groups = get_group_tensors(train_non_ctrl)
    eval_groups = get_group_tensors(eval_non_ctrl)
    val_groups = get_group_tensors(val_adata[val_adata.obs[pert_col].astype(str) != control_label]) if val_adata is not None else None

    # 4. 构建模型与优化器
    use_dose_flag = bool(dose_col and dose_col in adata.obs.columns)
    model = BioFlowCellNet(
        n_genes=n_genes,
        n_tokens=len(vocab_tokens),
        prior_dim=prior_matrix.shape[1],
        hidden_dim=int(hidden_dim),
        n_blocks=int(n_blocks),
        dropout=float(dropout),
        use_dose=use_dose_flag,
        prior_matrix=prior_matrix,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=float(lr) * 0.05)

    best_val_score = -1e9
    best_state = None
    t_start = time.time()

    # 预先把 Control pool 和训练集细胞常驻 GPU，彻底消除 CPU->GPU 拷贝瓶颈
    train_ctrl_gpu = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in train_ctrl_pools.items()}
    train_ctrl_mean_gpu = {k: v.mean(dim=0) for k, v in train_ctrl_gpu.items()}
    for g in train_groups:
        X_sub = g["X"]
        if X_sub.shape[0] > 64:
            idx_keep = np.random.choice(X_sub.shape[0], size=64, replace=False)
            X_sub = X_sub[idx_keep]
        g["X_gpu"] = torch.tensor(X_sub, device=device, dtype=torch.float32)
        g["mu_pert_gpu"] = torch.tensor(g["X"].mean(axis=0), device=device, dtype=torch.float32)
        g["tids_gpu"] = torch.tensor(g["token_ids"], device=device, dtype=torch.long)
        g["dlin_gpu"] = torch.tensor(g["delta_lin"], device=device, dtype=torch.float32)

    def eval_on_groups(groups_list, ctrl_pools_dict):
        model.eval()
        true_deltas = {}
        pred_deltas = {}
        with torch.no_grad():
            for g in groups_list:
                c_pool = ctrl_pools_dict.get(g["ctx"], ctrl_pools_dict["__ALL__"])
                mu_c = c_pool.mean(axis=0)
                true_deltas[g["pert"]] = g["X"].mean(axis=0) - mu_c

                # 采样最多 64 个 control 细胞进行 ODE 分布推演
                n_s = min(64, len(c_pool))
                c_idx = np.random.choice(len(c_pool), size=n_s, replace=False)
                x0 = torch.tensor(c_pool[c_idx], device=device, dtype=torch.float32)
                if center_by_context:
                    mu_c_t = torch.tensor(mu_c, device=device, dtype=torch.float32).unsqueeze(0)
                    x0_in = x0 - mu_c_t
                else:
                    x0_in = x0
                tids = torch.tensor(g["token_ids"], device=device, dtype=torch.long).unsqueeze(0).expand(n_s, -1)
                d_lin = torch.tensor(g["delta_lin"], device=device, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
                d_val = torch.full((n_s,), g["dose"], device=device, dtype=torch.float32) if use_dose_flag else None

                x1_pred = model.sample_ode(x0_in, tids, d_lin, dose_vals=d_val, steps=int(ode_steps))
                if center_by_context:
                    pred_delta = x1_pred.mean(dim=0).cpu().numpy()
                else:
                    pred_delta = (x1_pred.mean(dim=0).cpu().numpy()) - mu_c
                pred_deltas[g["pert"]] = pred_delta

        df_p = evaluate_perturbation_predictions(true_deltas, pred_deltas, model_name, top_k_list)
        return df_p

    # 5. 训练主循环（按扰动组 mini-batch + 混合精度）
    n_groups = len(train_groups)
    steps_per_epoch = max(1, math.ceil(n_groups / int(groups_per_step)))

    for ep in range(1, int(epochs) + 1):
        model.train()
        perm = np.random.permutation(n_groups)
        ep_loss = 0.0

        for st in range(steps_per_epoch):
            batch_g_ids = perm[st * int(groups_per_step) : (st + 1) * int(groups_per_step)]
            optimizer.zero_grad()

            loss_cfm_acc = 0.0
            loss_cos_acc = 0.0
            loss_huber_acc = 0.0
            loss_mse_acc = 0.0
            cnt_g = 0
            gate_vec = model.get_prior_gate()

            for gid in batch_g_ids:
                g = train_groups[gid]
                c_pool_t = train_ctrl_gpu.get(g["ctx"], train_ctrl_gpu["__ALL__"])
                c_mean_t = train_ctrl_mean_gpu.get(g["ctx"], train_ctrl_mean_gpu["__ALL__"])
                n_c = int(cells_per_group_batch)

                c_idx = torch.randint(0, c_pool_t.shape[0], (n_c,), device=device)
                p_idx = torch.randint(0, g["X_gpu"].shape[0], (n_c,), device=device)

                x0 = c_pool_t[c_idx]
                x1 = g["X_gpu"][p_idx]
                if center_by_context:
                    x0 = x0 - c_mean_t.unsqueeze(0)
                    x1 = x1 - c_mean_t.unsqueeze(0)
                d_lin = g["dlin_gpu"].unsqueeze(0).expand(n_c, -1)
                tids = g["tids_gpu"].unsqueeze(0).expand(n_c, -1)
                d_val = torch.full((n_c,), g["dose"], device=device, dtype=torch.float32) if use_dose_flag else None

                # 残差流匹配目标: 把 x0 推向 (x1 - gated_linear_prior)
                x1_res = x1 - d_lin * gate_vec
                t = torch.rand((n_c,), device=device, dtype=torch.float32)
                t_col = t.unsqueeze(-1)
                x_t = (1.0 - t_col) * x0 + t_col * x1_res
                u_t = x1_res - x0

                v_pred = model(x_t, x0, t, tids, d_val)
                loss_cfm_acc = loss_cfm_acc + F.mse_loss(v_pred, u_t)

                # 一步残差还原做伪批量方向与幅度双校准 (使用全组真实均值降低采样方差)
                v_step0 = model(x0, x0, torch.zeros((n_c,), device=device), tids, d_val)
                pred_delta_grp = (v_step0 + d_lin * gate_vec).mean(dim=0)
                true_delta_grp = g["mu_pert_gpu"] - c_mean_t

                cos_sim = F.cosine_similarity(pred_delta_grp.unsqueeze(0), true_delta_grp.unsqueeze(0)).mean()
                loss_cos_acc = loss_cos_acc + (1.0 - cos_sim)
                loss_huber_acc = loss_huber_acc + F.smooth_l1_loss(pred_delta_grp, true_delta_grp, beta=0.1)
                loss_mse_acc = loss_mse_acc + F.mse_loss(pred_delta_grp, true_delta_grp)
                cnt_g += 1

            total_loss = (
                float(lambda_cfm) * (loss_cfm_acc / cnt_g)
                + float(lambda_cos) * (loss_cos_acc / cnt_g)
                + float(lambda_huber) * (loss_huber_acc / cnt_g)
                + float(lambda_mse) * (loss_mse_acc / cnt_g)
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_loss += float(total_loss.item())

        scheduler.step()

        # 周期性验证与保存最优权重（严格防泄漏：无独立 val 集时不触碰 test 集）
        if ep % 5 == 0 or ep == int(epochs):
            if val_groups is not None:
                val_df = eval_on_groups(val_groups, val_ctrl_pools)
                val_r = float(val_df["pearson_delta_all"].mean())
                val_top20 = float(val_df[f"pearson_delta_top{top_k_list[0]}"].mean())
                print(
                    f"[{model_name}] Epoch {ep:03d}/{epochs} | Loss={ep_loss/steps_per_epoch:.4f} | "
                    f"Val Pearson={val_r:.4f} | Val Top{top_k_list[0]}={val_top20:.4f} | Time={time.time()-t_start:.1f}s"
                )
                score = val_r + 0.5 * val_top20
                if score > best_val_score:
                    best_val_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                print(
                    f"[{model_name}] Epoch {ep:03d}/{epochs} | Loss={ep_loss/steps_per_epoch:.4f} | Time={time.time()-t_start:.1f}s"
                )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if save_ckpt_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_ckpt_path)), exist_ok=True)
        torch.save(model.state_dict(), save_ckpt_path)
        print(f"[{model_name}] 最优模型权重已保存至: {save_ckpt_path}")

    # 6. 最终测试集评测
    per_pert_df = eval_on_groups(eval_groups, eval_ctrl_pools)
    metric_cols = [c for c in per_pert_df.columns if c not in ("model", "perturbation")]
    srow = {"model": model_name, "n_eval_perts": int(len(per_pert_df))}
    for mc in metric_cols:
        srow[f"{mc}_mean"] = float(per_pert_df[mc].mean())
        srow[f"{mc}_std"] = float(per_pert_df[mc].std())
    summary_df = pd.DataFrame([srow])

    print(f"\n[{model_name}] 最终测试集 ({test_split}) 表现:")
    print(
        f"  Pearson(Delta) All = {srow['pearson_delta_all_mean']:.4f} | "
        f"Top{top_k_list[0]} = {srow[f'pearson_delta_top{top_k_list[0]}_mean']:.4f} | "
        f"DirAcc Top{top_k_list[0]} = {srow[f'dir_acc_top{top_k_list[0]}_mean']:.4f} | "
        f"MSE = {srow['mse_all_mean']:.6f}"
    )

    if per_pert_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(per_pert_csv_path)), exist_ok=True)
        per_pert_df.to_csv(per_pert_csv_path, index=False, encoding="utf-8")
    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    return summary_df, per_pert_df


# 中文注释：BioFlowCell 原创虚拟细胞模型实现，包含线性残差流匹配、基因共表达/教师先验融合、Set-Attention 二阶交互与细胞基态语境编码。
