# -*- coding: utf-8 -*-
"""
Geneformer-V2-104M_CLcancer 大基座特征提取与知识蒸馏模块 (Geneformer Distillation)
支持：
1. 提取 Geneformer 768 维基因静态先验嵌入；
2. 利用 Geneformer Transformer (FP16) 提取细胞级上下文表征，作为 Teacher 隐空间知识；
3. 训练轻量级知识蒸馏学生模型 DistillCell，并输出与线性基线同口径的评测指标。
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


def extract_geneformer_gene_embeddings(
    geneformer_dir: str,
    token_dict_pkl: str,
    gene_name_id_pkl: str,
    target_genes: list,
    var_ensembl_ids: dict = None,
) -> dict:
    """
    从 Geneformer-V2-104M_CLcancer 的 word_embeddings 提取目标基因的 768 维预训练嵌入。
    """
    with open(token_dict_pkl, "rb") as f:
        tok_dict = pickle.load(f)
    with open(gene_name_id_pkl, "rb") as f:
        name_to_ens = pickle.load(f)

    gf_model = BertModel.from_pretrained(geneformer_dir)
    emb_weight = gf_model.embeddings.word_embeddings.weight.detach().cpu().numpy().astype(np.float32)
    del gf_model

    gene_emb_dict = {}
    matched = 0
    for g in target_genes:
        ens = None
        if var_ensembl_ids and g in var_ensembl_ids:
            ens = str(var_ensembl_ids[g])
        elif g in name_to_ens:
            ens = str(name_to_ens[g])

        if ens and ens in tok_dict:
            tid = int(tok_dict[ens])
            vec = emb_weight[tid].copy()
            norm = np.linalg.norm(vec) + 1e-6
            gene_emb_dict[g] = vec / norm
            matched += 1

    print(f"[GF-Distill] 成功匹配 Geneformer 基因嵌入: {matched}/{len(target_genes)} 个基因 (dim={emb_weight.shape[1]})")
    return gene_emb_dict


@torch.no_grad()
def extract_geneformer_cell_teacher_embeddings(
    adata,
    geneformer_dir: str,
    token_dict_pkl: str,
    gene_name_id_pkl: str,
    ensembl_col: str = "ensemble_id",
    max_seq_len: int = 512,
    batch_size: int = 32,
    max_cells_per_group: int = 16,
    pert_col: str = "perturbation",
    split_col: str = "ood_split",
    target_split: str = "train",
) -> dict:
    """
    使用 Geneformer-V2-104M_CLcancer (FP16) 对训练集每个扰动组提取细胞级上下文 Teacher 表征 (768维)。
    按细胞内表达量降序排列基因 Token（Geneformer 标准 Rank-Value Encoding 近似）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(token_dict_pkl, "rb") as f:
        tok_dict = pickle.load(f)
    with open(gene_name_id_pkl, "rb") as f:
        name_to_ens = pickle.load(f)

    cls_id = tok_dict.get("<cls>", 2)
    pad_id = tok_dict.get("<pad>", 0)

    # 映射 adata.var_names -> Geneformer token_id
    col_token_ids = np.zeros(adata.n_vars, dtype=np.int64)
    has_ens_col = ensembl_col in adata.var.columns
    for j, g in enumerate(adata.var_names):
        ens = str(adata.var[ensembl_col].iloc[j]) if has_ens_col else name_to_ens.get(g, "")
        if ens in tok_dict:
            col_token_ids[j] = int(tok_dict[ens])

    valid_gene_mask = col_token_ids > 0
    valid_col_indices = np.where(valid_gene_mask)[0]
    valid_tids = col_token_ids[valid_col_indices]
    print(f"[GF-Distill] 细胞表征编码对齐基因数: {len(valid_col_indices)}/{adata.n_vars}")

    gf_model = BertModel.from_pretrained(geneformer_dir, torch_dtype=torch.float16).to(device)
    gf_model.eval()

    sub = adata[adata.obs[split_col] == target_split]
    perts = sub.obs[pert_col].astype(str).values
    df_idx = pd.DataFrame({"p": perts, "row": np.arange(len(perts))})

    pert_teacher_emb = {}
    all_seqs = []
    all_pert_keys = []

    for p_name, grp in df_idx.groupby("p", sort=True):
        rows = grp["row"].values
        if len(rows) > max_cells_per_group:
            rows = np.random.choice(rows, size=max_cells_per_group, replace=False)
        X_sub = _to_dense_array(sub.X[rows])[:, valid_col_indices]
        for r in range(X_sub.shape[0]):
            vals = X_sub[r]
            nz = np.where(vals > 0)[0]
            if len(nz) == 0:
                seq = [cls_id]
            else:
                order = nz[np.argsort(-vals[nz])[: max_seq_len - 1]]
                seq = [cls_id] + valid_tids[order].tolist()
            all_seqs.append(seq)
            all_pert_keys.append(p_name)

    # 批量推理 Teacher 隐藏层表征
    embs_list = []
    for i in range(0, len(all_seqs), batch_size):
        batch_seqs = all_seqs[i : i + batch_size]
        max_l = max(len(s) for s in batch_seqs)
        inp = torch.full((len(batch_seqs), max_l), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(batch_seqs), max_l), dtype=torch.long, device=device)
        for b, s in enumerate(batch_seqs):
            inp[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[b, : len(s)] = 1
        out = gf_model(input_ids=inp, attention_mask=attn)
        # 取 [CLS] 表征作为细胞上下文 Teacher 向量
        cls_vec = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        embs_list.append(cls_vec)

    del gf_model
    torch.cuda.empty_cache()

    all_embs = np.vstack(embs_list)
    df_emb = pd.DataFrame({"p": all_pert_keys, "idx": np.arange(len(all_pert_keys))})
    for p_name, grp in df_emb.groupby("p"):
        mean_vec = all_embs[grp["idx"].values].mean(axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-6)
        pert_teacher_emb[p_name] = mean_vec.astype(np.float32)

    print(f"[GF-Distill] 已提取 {len(pert_teacher_emb)} 个扰动组的 Geneformer Teacher 细胞上下文表征")
    return pert_teacher_emb


class DistillStudentNet(nn.Module):
    """
    轻量级知识蒸馏学生模型 (DistillCell):
    输入细胞基态 + Geneformer 基因先验，通过中间层特征蒸馏对齐 Geneformer 768维上下文表征，
    并叠加线性残差预测扰动偏移 Delta。
    """
    def __init__(self, n_genes: int, n_tokens: int, gf_prior_mat: np.ndarray, hidden_dim: int = 256, teacher_dim: int = 768):
        super().__init__()
        self.register_buffer("gf_prior", torch.tensor(gf_prior_mat, dtype=torch.float32))
        self.token_emb = nn.Embedding(n_tokens + 1, 64, padding_idx=0)
        self.gf_proj = nn.Linear(gf_prior_mat.shape[1], hidden_dim - 64)
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
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # 蒸馏投影头：对齐 Geneformer Teacher 768 维表征
        self.distill_head = nn.Linear(hidden_dim, teacher_dim)
        # 表达残差预测头
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.prior_gate_raw = nn.Parameter(torch.zeros(1, n_genes))

    def get_prior_gate(self) -> torch.Tensor:
        return 1.0 + 0.25 * torch.tanh(self.prior_gate_raw)

    def forward(self, x_ctrl_mean: torch.Tensor, token_ids: torch.Tensor, delta_lin: torch.Tensor):
        mask = (token_ids == 0)
        valid = (~mask).unsqueeze(-1).float()
        e_tok = self.token_emb(token_ids)
        e_gf = self.gf_proj(self.gf_prior[token_ids])
        h_tok = self.pert_enc(torch.cat([e_tok, e_gf], dim=-1))
        h_sum = (h_tok * valid).sum(dim=1)
        if token_ids.shape[1] >= 2:
            h_pair = (h_tok[:, 0, :] * h_tok[:, 1, :]) * (valid[:, 0, :] * valid[:, 1, :])
        else:
            h_pair = torch.zeros_like(h_sum)
        z_ctx = self.ctx_enc((x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6))
        feat = self.fusion(torch.cat([z_ctx, h_sum, h_pair], dim=-1))
        t_pred = F.normalize(self.distill_head(feat), dim=-1)
        delta_pred = delta_lin * self.get_prior_gate() + self.delta_head(feat)
        return delta_pred, t_pred


def train_and_eval_distill_student(
    adata_or_path,
    geneformer_dir: str,
    token_dict_pkl: str,
    gene_name_id_pkl: str,
    pert_col: str = "perturbation",
    control_label: str = "control",
    sep_char: str = "_",
    split_col: str = "ood_split",
    context_col: str = "",
    ensembl_col: str = "ensemble_id",
    ridge_alpha: float = 10.0,
    hidden_dim: int = 256,
    epochs: int = 80,
    lr: float = 1e-3,
    lambda_distill: float = 0.5,
    lambda_mse: float = 25.0,
    top_k_str: str = "20;50",
    model_name: str = "DistillCell-GF",
    save_ckpt_path: str = "",
    per_pert_csv_path: str = "",
    summary_csv_path: str = "",
):
    """训练并评测 Geneformer 知识蒸馏学生模型 DistillCell。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k_list = tuple(int(x.strip()) for x in str(top_k_str).split(";") if x.strip())

    adata = sc.read_h5ad(adata_or_path) if isinstance(adata_or_path, str) else adata_or_path
    t0 = time.time()

    # 1. 提取 Teacher 静态基因嵌入 + 细胞上下文表征
    all_non_ctrl = sorted(set(adata.obs.loc[adata.obs[pert_col] != control_label, pert_col].astype(str)))
    def parse_toks(p_str):
        base_p = p_str.split("__dose_")[0] if "__dose_" in p_str else p_str
        return [t.strip() for t in base_p.split(sep_char) if t.strip()]

    vocab_tokens = sorted({tok for p in all_non_ctrl for tok in parse_toks(p)})
    token_to_id = {tok: i + 1 for i, tok in enumerate(vocab_tokens)}
    max_len = max(len(parse_toks(p)) for p in all_non_ctrl)

    var_ens = {g: str(adata.var[ensembl_col].iloc[i]) for i, g in enumerate(adata.var_names)} if ensembl_col in adata.var.columns else None
    gf_gene_embs = extract_geneformer_gene_embeddings(geneformer_dir, token_dict_pkl, gene_name_id_pkl, vocab_tokens, var_ens)
    t_dim = len(next(iter(gf_gene_embs.values()))) if len(gf_gene_embs) > 0 else 768
    gf_prior_mat = np.zeros((len(vocab_tokens) + 1, t_dim), dtype=np.float32)
    for tok, tid in token_to_id.items():
        if tok in gf_gene_embs:
            gf_prior_mat[tid] = gf_gene_embs[tok]

    teacher_cell_embs = extract_geneformer_cell_teacher_embeddings(
        adata, geneformer_dir, token_dict_pkl, gene_name_id_pkl,
        ensembl_col=ensembl_col, pert_col=pert_col, split_col=split_col, target_split="train"
    )

    # 2. 准备训练集与测试集组均值及线性先验
    train_adata = adata[adata.obs[split_col] == "train"]
    val_adata = adata[adata.obs[split_col] == "val"] if (adata.obs[split_col] == "val").sum() > 0 else None
    eval_adata = adata[adata.obs[split_col] == "test"]

    def get_ctrl_mean(sub):
        cm = (sub.obs[pert_col].astype(str) == control_label).values
        return _to_dense_array(sub.X[cm]).mean(axis=0)

    mu_ctrl_train = get_ctrl_mean(train_adata)
    mu_ctrl_eval = get_ctrl_mean(eval_adata)
    mu_ctrl_val = get_ctrl_mean(val_adata) if val_adata is not None else None

    train_deltas = _compute_group_deltas(train_adata[train_adata.obs[pert_col] != control_label], pert_col, mu_ctrl_train)
    eval_deltas = _compute_group_deltas(eval_adata[eval_adata.obs[pert_col] != control_label], pert_col, mu_ctrl_eval)
    val_deltas = _compute_group_deltas(val_adata[val_adata.obs[pert_col] != control_label], pert_col, mu_ctrl_val) if val_adata is not None else None

    train_vocab = sorted({t for p in train_deltas for t in parse_toks(p)})
    lin_idx = {t: i for i, t in enumerate(train_vocab)}
    def encode_lin(plist):
        m = np.zeros((len(plist), len(train_vocab)), dtype=np.float32)
        for ri, ps in enumerate(plist):
            for t in parse_toks(ps):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    tr_keys = sorted(train_deltas.keys())
    ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    ridge.fit(encode_lin(tr_keys), np.vstack([train_deltas[p] for p in tr_keys]))

    def build_tensors(delta_dict, mu_c):
        p_keys = sorted(delta_dict.keys())
        tids = np.zeros((len(p_keys), max_len), dtype=np.int64)
        for i, p in enumerate(p_keys):
            for j, t in enumerate(parse_toks(p)[:max_len]):
                tids[i, j] = token_to_id.get(t, 0)
        d_lin = ridge.predict(encode_lin(p_keys)).astype(np.float32)
        y_true = np.vstack([delta_dict[p] for p in p_keys]).astype(np.float32)
        x_c = np.tile(mu_c, (len(p_keys), 1)).astype(np.float32)
        t_embs = np.vstack([teacher_cell_embs.get(p, np.zeros(t_dim, dtype=np.float32)) for p in p_keys])
        return (
            p_keys,
            torch.tensor(x_c, device=device),
            torch.tensor(tids, device=device),
            torch.tensor(d_lin, device=device),
            torch.tensor(y_true, device=device),
            torch.tensor(t_embs, device=device),
        )

    tr_p, tr_xc, tr_tid, tr_dlin, tr_y, tr_temb = build_tensors(train_deltas, mu_ctrl_train)
    ev_p, ev_xc, ev_tid, ev_dlin, ev_y, _ = build_tensors(eval_deltas, mu_ctrl_eval)
    if val_deltas is not None:
        va_p, va_xc, va_tid, va_dlin, va_y, _ = build_tensors(val_deltas, mu_ctrl_val)

    student = DistillStudentNet(adata.n_vars, len(vocab_tokens), gf_prior_mat, hidden_dim=hidden_dim, teacher_dim=t_dim).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    best_score = -1e9
    best_state = None
    for ep in range(1, epochs + 1):
        student.train()
        opt.zero_grad()
        pred_d, pred_t = student(tr_xc, tr_tid, tr_dlin)
        loss_huber = F.smooth_l1_loss(pred_d, tr_y, beta=0.1)
        loss_mse = F.mse_loss(pred_d, tr_y)
        loss_cos = (1.0 - F.cosine_similarity(pred_d, tr_y, dim=-1)).mean()
        loss_distill = (1.0 - F.cosine_similarity(pred_t, tr_temb, dim=-1)).mean()
        loss = loss_huber + float(lambda_mse) * loss_mse + 0.8 * loss_cos + lambda_distill * loss_distill
        loss.backward()
        opt.step()
        sched.step()

        if ep % 10 == 0 or ep == epochs:
            if val_deltas is not None:
                student.eval()
                with torch.no_grad():
                    v_pred, _ = student(va_xc, va_tid, va_dlin)
                    v_np = v_pred.cpu().numpy()
                    df_v = evaluate_perturbation_predictions({p: val_deltas[p] for p in va_p}, {p: v_np[i] for i, p in enumerate(va_p)}, model_name, top_k_list)
                    vr = float(df_v["pearson_delta_all"].mean())
                    vt20 = float(df_v[f"pearson_delta_top{top_k_list[0]}"].mean())
                    vmse = float(df_v["mse_all"].mean())
                    print(f"[{model_name}] Epoch {ep:03d}/{epochs} | Loss={loss.item():.4f} | Val Pearson={vr:.4f} | Val Top{top_k_list[0]}={vt20:.4f} | Val MSE={vmse:.6f}")
                    if vr + 0.5 * vt20 - 10.0 * vmse > best_score:
                        best_score = vr + 0.5 * vt20 - 10.0 * vmse
                        best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            else:
                print(f"[{model_name}] Epoch {ep:03d}/{epochs} | Loss={loss.item():.4f}")

    if best_state is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if save_ckpt_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_ckpt_path)), exist_ok=True)
        torch.save(student.state_dict(), save_ckpt_path)

    student.eval()
    with torch.no_grad():
        e_pred, _ = student(ev_xc, ev_tid, ev_dlin)
        e_np = e_pred.cpu().numpy()
    per_pert_df = evaluate_perturbation_predictions(eval_deltas, {p: e_np[i] for i, p in enumerate(ev_p)}, model_name, top_k_list)

    metric_cols = [c for c in per_pert_df.columns if c not in ("model", "perturbation")]
    srow = {"model": model_name, "n_eval_perts": int(len(per_pert_df))}
    for mc in metric_cols:
        srow[f"{mc}_mean"] = float(per_pert_df[mc].mean())
        srow[f"{mc}_std"] = float(per_pert_df[mc].std())
    summary_df = pd.DataFrame([srow])

    print(
        f"\n[{model_name}] 测试集表现: Pearson All={srow['pearson_delta_all_mean']:.4f} | "
        f"Top{top_k_list[0]}={srow[f'pearson_delta_top{top_k_list[0]}_mean']:.4f} | "
        f"MSE={srow['mse_all_mean']:.6f} | 耗时={time.time()-t0:.1f}s"
    )

    if per_pert_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(per_pert_csv_path)), exist_ok=True)
        per_pert_df.to_csv(per_pert_csv_path, index=False, encoding="utf-8")
    if summary_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

    return summary_df, per_pert_df, gf_gene_embs


# 中文注释：本模块实现基于 Geneformer-V2-104M_CLcancer 开源单细胞大模型的基因与细胞上下文特征提取及知识蒸馏学生模型 DistillCell。
