# -*- coding: utf-8 -*-
"""
pgnn.py - 全量人类通路“基因-通路二部图”底座模块 (Pathway-Gene Bipartite GNN)
实现：
  - 第 1 层：OmniPath 有向激活(+1)/抑制(-1) + BioGRID 局部物理邻域残差消息传递
  - 第 2 层：全量人类通路超节点并行注意力汇集与广播 (Gene <-> Pathway)
  - 未知/蛋白修饰缺失节点 [MASK] 机制：对未观测或被掩码的激酶/信号节点注入可学习潜状态，由上下游因果足迹双向推断
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_norm_dense(mat: np.ndarray) -> np.ndarray:
    """对二维非负邻接矩阵按行做对称/出度归一化，避免度数大节点数值爆炸。"""
    rs = mat.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return (mat / rs).astype(np.float32)


def build_gene_pathway_bipartite_tensors(
    target_genes: list,
    priors_dir: str,
    min_genes_per_pathway: int = 3,
    include_biogrid: bool = True,
) -> dict:
    """
    将全量先验图谱对齐到目标基因列表，输出有向激活/抑制矩阵、BioGRID 矩阵及基因-通路隶属矩阵。
    """
    n_genes = len(target_genes)
    gene_upper = [str(g).strip().upper() for g in target_genes]
    g2i = {g: i for i, g in enumerate(gene_upper)}

    # 1. OmniPath 有向激活(+1)/抑制(-1)与双向无符号边
    dir_path = os.path.join(priors_dir, "directed_signaling_edges.parquet")
    dir_df = pd.read_parquet(dir_path)
    A_pos = np.zeros((n_genes, n_genes), dtype=np.float32)
    A_neg = np.zeros((n_genes, n_genes), dtype=np.float32)
    A_und = np.zeros((n_genes, n_genes), dtype=np.float32)

    n_pos_hit, n_neg_hit, n_und_hit = 0, 0, 0
    for src, dst, sgn in zip(dir_df["src"], dir_df["dst"], dir_df["sign"]):
        i = g2i.get(str(src).upper())
        j = g2i.get(str(dst).upper())
        if i is None or j is None or i == j:
            continue
        # j 接收来自 i 的信号 -> 行 j 聚合列 i
        if sgn == 1:
            A_pos[j, i] = 1.0
            n_pos_hit += 1
        elif sgn == -1:
            A_neg[j, i] = 1.0
            n_neg_hit += 1
        else:
            A_und[j, i] = 1.0
            n_und_hit += 1

    # 2. BioGRID 物理蛋白互作作为局部对称补充骨架
    n_bio_hit = 0
    bio_path = os.path.join(priors_dir, "biogrid_human_ppi.parquet")
    if include_biogrid and os.path.exists(bio_path):
        bio_df = pd.read_parquet(bio_path)
        for ga, gb in zip(bio_df["gene_a"], bio_df["gene_b"]):
            ia = g2i.get(str(ga).upper())
            ib = g2i.get(str(gb).upper())
            if ia is not None and ib is not None and ia != ib:
                A_und[ia, ib] = 1.0
                A_und[ib, ia] = 1.0
                n_bio_hit += 1

    # 3. 全量人类通路隶属矩阵 H_gp: (n_genes, n_pathways)
    mem_path = os.path.join(priors_dir, "gene_pathway_membership.parquet")
    mem_df = pd.read_parquet(mem_path)
    mem_df["g_up"] = mem_df["gene_symbol"].astype(str).str.upper()
    mem_sub = mem_df[mem_df["g_up"].isin(g2i)].copy()

    pw_counts = mem_sub.groupby("pathway_id")["g_up"].nunique()
    valid_pws = sorted(pw_counts[pw_counts >= int(min_genes_per_pathway)].index.tolist())
    pw2idx = {p: k for k, p in enumerate(valid_pws)}
    H_gp = np.zeros((n_genes, len(valid_pws)), dtype=np.float32)
    for g_up, pw_id in zip(mem_sub["g_up"], mem_sub["pathway_id"]):
        k = pw2idx.get(pw_id)
        if k is not None:
            H_gp[g2i[g_up], k] = 1.0

    # 归一化 Gene->Pathway (按通路大小) 与 Pathway->Gene (按基因所属通路数)
    pw_deg = np.maximum(H_gp.sum(axis=0, keepdims=True), 1.0)
    g_deg = np.maximum(H_gp.sum(axis=1, keepdims=True), 1.0)
    H_g2p = (H_gp / pw_deg).astype(np.float32)       # (n_genes, n_pws)
    H_p2g = (H_gp / g_deg).T.astype(np.float32)      # (n_pws, n_genes)

    # 识别关键激酶/信号枢纽掩码候选 (用于蛋白修饰缺失压力测试)
    # 选取既发出有向信号又接收有向信号的高介数信号节点
    out_deg = (A_pos.sum(axis=0) + A_neg.sum(axis=0))
    in_deg = (A_pos.sum(axis=1) + A_neg.sum(axis=1))
    hub_score = out_deg * in_deg
    kinase_like_mask = (hub_score > 0).astype(np.float32)

    print(
        f"[PGNN] 先验对齐完成: 基因={n_genes}, 激活边={n_pos_hit}, 抑制边={n_neg_hit}, "
        f"BioGRID边={n_bio_hit}, 覆盖通路={len(valid_pws)}, 信号枢纽节点={int(kinase_like_mask.sum())}"
    )

    return {
        "A_pos": _row_norm_dense(A_pos),
        "A_neg": _row_norm_dense(A_neg),
        "A_und": _row_norm_dense(A_und),
        "H_g2p": H_g2p,
        "H_p2g": H_p2g,
        "pathway_ids": valid_pws,
        "kinase_like_mask": kinase_like_mask,
    }


class PathwayGeneBipartiteNet(nn.Module):
    """
    两层“基因-通路二部图”底座网络 + 未知/蛋白修饰缺失节点 [MASK] 推断机制。
    在保留 v1.0 Cross-Lineage Ridge 与靶基因基态门控的基础上，注入：
      - 第 1 层：有向激活(+)/抑制(-) + 局部互作的 1 跳残差消息传递（防过平滑）
      - 第 2 层：全量通路超节点并行注意力汇集与基因回传（Gene <-> Pathway）
      - 未知状态节点显式使用可学习 [MASK] embedding，由上下游信号反推状态
    """

    def __init__(
        self,
        n_genes: int,
        n_teacher: int,
        U_prog: np.ndarray,
        bipartite_tensors: dict,
        d_node: int = 32,
        hid: int = 128,
        use_latent_mask: bool = False,
        mask_prob_train: float = 0.0,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.d_node = int(d_node)
        self.use_latent_mask = bool(use_latent_mask)
        self.mask_prob_train = float(mask_prob_train)

        # 低秩程序基与有向/二部先验矩阵
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.register_buffer("A_pos", torch.tensor(bipartite_tensors["A_pos"], dtype=torch.float32))
        self.register_buffer("A_neg", torch.tensor(bipartite_tensors["A_neg"], dtype=torch.float32))
        self.register_buffer("A_und", torch.tensor(bipartite_tensors["A_und"], dtype=torch.float32))
        self.register_buffer("H_g2p", torch.tensor(bipartite_tensors["H_g2p"], dtype=torch.float32))
        self.register_buffer("H_p2g", torch.tensor(bipartite_tensors["H_p2g"], dtype=torch.float32))
        self.register_buffer(
            "kinase_like_mask",
            torch.tensor(bipartite_tensors["kinase_like_mask"], dtype=torch.float32),
        )
        k_prog = U_prog.shape[0]
        n_pws = self.H_g2p.shape[1]

        # 保留 v1.0 靶基因基态表达门控与 Ridge 先验缩放
        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.target_gate = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)

        # 可学习未知/蛋白磷酸化缺失节点 [MASK] 向量与基因节点编码
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_node) * 0.02)
        self.node_in = nn.Linear(2, d_node)  # [基态表达标量, 是否为受扰动靶点指示]
        self.gene_id_emb = nn.Parameter(torch.randn(1, n_genes, d_node) * 0.02)

        # 第 1 层：有向激活/抑制 + 无向局部残差消息传递
        self.msg_pos = nn.Linear(d_node, d_node, bias=False)
        self.msg_neg = nn.Linear(d_node, d_node, bias=False)
        self.msg_und = nn.Linear(d_node, d_node, bias=False)
        self.layer1_norm = nn.LayerNorm(d_node)

        # 第 2 层：全量通路超节点并行汇集 (Gene -> Pathway) 与回传 (Pathway -> Gene)
        self.pw_attn_gate = nn.Linear(d_node, 1)
        self.pw_mlp = nn.Sequential(nn.Linear(d_node, d_node), nn.GELU())
        self.layer2_norm = nn.LayerNorm(d_node)

        # 全通路汇集特征 -> 基因程序修正系数
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.gnn_to_prog = nn.Sequential(
            nn.Linear(n_pws + hid + 3, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, k_prog),
        )
        # 节点级直通残差（避免过度汇聚稀释单基因靶点尖峰信号）
        self.node_readout = nn.Linear(d_node, 1)
        nn.init.zeros_(self.gnn_to_prog[-1].weight)
        nn.init.zeros_(self.gnn_to_prog[-1].bias)
        nn.init.zeros_(self.node_readout.weight)
        nn.init.zeros_(self.node_readout.bias)

    def forward(
        self,
        x_ctrl: torch.Tensor,
        x_teach: torch.Tensor,
        x_ridge: torch.Tensor,
        x_tstat: torch.Tensor,
        pert_onehot: torch.Tensor,
        force_mask_kinases: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_ctrl.shape[0]
        # 组装节点初始输入: (B, G, 2)
        node_feat = torch.stack([x_ctrl, pert_onehot], dim=-1)
        h0 = self.node_in(node_feat) + self.gene_id_emb  # (B, G, d)

        # 缺失/蛋白修饰掩码处理：用可学习 [MASK] token 替代被掩码节点，迫使图网络从上下游推演
        if self.use_latent_mask:
            if force_mask_kinases:
                m = self.kinase_like_mask.view(1, self.n_genes, 1)
                h0 = h0 * (1.0 - m) + self.mask_token * m
            elif self.training and self.mask_prob_train > 0.0:
                rand_m = (torch.rand(B, self.n_genes, 1, device=h0.device) < self.mask_prob_train).float()
                m = rand_m * self.kinase_like_mask.view(1, self.n_genes, 1)
                h0 = h0 * (1.0 - m) + self.mask_token * m

        # 第 1 层：1 跳有向激活(+) / 抑制(-) / 局部互作消息传递 (直接用 matmul 触发 cuBLAS 加速)
        m_pos = torch.matmul(self.A_pos, self.msg_pos(h0))
        m_neg = torch.matmul(self.A_neg, self.msg_neg(h0))
        m_und = torch.matmul(self.A_und, self.msg_und(h0))
        h1 = self.layer1_norm(h0 + F.gelu(m_pos - m_neg + 0.5 * m_und))  # (B, G, d)

        # 第 2 层：全量人类通路超节点并行注意力汇集 (Gene -> Pathway)
        node_imp = torch.sigmoid(self.pw_attn_gate(h1))  # (B, G, 1)
        h1_weighted = h1 * node_imp
        p_state = torch.matmul(self.H_g2p.t(), h1_weighted)  # (B, P, d)
        p_state = self.pw_mlp(p_state)
        pw_activity = p_state.mean(dim=-1)  # (B, P) 每条通路的即时活性读数

        # 通路高阶信号回传至基因节点 + 残差连接 (Pathway -> Gene)
        h_from_pw = torch.matmul(self.H_p2g.t(), p_state)
        h2 = self.layer2_norm(h1 + 0.5 * h_from_pw)

        # 解码：靶基因门控 Ridge 先验 + 通路程序残差 + 基因节点局部残差
        gate = 1.0 + 0.5 * torch.tanh(self.target_gate(x_tstat))
        h_joint = torch.cat([pw_activity, self.teach_enc(x_teach), x_tstat], dim=-1)
        delta_prog = self.gnn_to_prog(h_joint) @ self.U_prog
        delta_local = self.node_readout(h2).squeeze(-1)

        pred_delta = gate * (self.ridge_scale * x_ridge) + 0.5 * delta_prog + 0.2 * delta_local
        return pred_delta, pw_activity


class PathwayGeneBipartiteNetV2(nn.Module):
    """
    Exp30b 净化版全通路二部图底座：
      1. 去掉直接扰动 3603 静止背景基因的 node_readout，严格在低秩基因程序空间解码（守住 MSE）
      2. 对 2665 条通路汇集活性做 32 维低秩瓶颈压缩 + Dropout(0.3)，砍掉 90% 过拟合参数
      3. 仅沿 OmniPath 有向激活(+)/抑制(-)边做 1 跳消息传递，避免稠密 BioGRID 噪声扩散
      4. 保留 [MASK] 激酶/信号节点掩码推断能力
    """

    def __init__(
        self,
        n_genes: int,
        n_teacher: int,
        U_prog: np.ndarray,
        bipartite_tensors: dict,
        d_node: int = 16,
        pw_bottleneck: int = 32,
        hid: int = 128,
        use_latent_mask: bool = False,
        mask_prob_train: float = 0.0,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.use_latent_mask = bool(use_latent_mask)
        self.mask_prob_train = float(mask_prob_train)

        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.register_buffer("A_pos", torch.tensor(bipartite_tensors["A_pos"], dtype=torch.float32))
        self.register_buffer("A_neg", torch.tensor(bipartite_tensors["A_neg"], dtype=torch.float32))
        self.register_buffer("H_g2p", torch.tensor(bipartite_tensors["H_g2p"], dtype=torch.float32))
        self.register_buffer(
            "kinase_like_mask",
            torch.tensor(bipartite_tensors["kinase_like_mask"], dtype=torch.float32),
        )
        k_prog = U_prog.shape[0]
        n_pws = self.H_g2p.shape[1]

        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.target_gate = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_node) * 0.02)
        self.node_in = nn.Linear(2, d_node)
        self.gene_id_emb = nn.Parameter(torch.randn(1, n_genes, d_node) * 0.02)

        self.msg_pos = nn.Linear(d_node, d_node, bias=False)
        self.msg_neg = nn.Linear(d_node, d_node, bias=False)
        self.layer1_norm = nn.LayerNorm(d_node)

        self.pw_attn_gate = nn.Linear(d_node, 1)
        self.pw_compress = nn.Sequential(
            nn.Linear(n_pws, pw_bottleneck),
            nn.LayerNorm(pw_bottleneck),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.prog_head = nn.Sequential(
            nn.Linear(64 + pw_bottleneck + hid + 3, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.prog_head[-1].weight)
        nn.init.zeros_(self.prog_head[-1].bias)

    def forward(
        self,
        x_ctrl: torch.Tensor,
        x_teach: torch.Tensor,
        x_ridge: torch.Tensor,
        x_tstat: torch.Tensor,
        pert_onehot: torch.Tensor,
        force_mask_kinases: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_ctrl.shape[0]
        h0 = self.node_in(torch.stack([x_ctrl, pert_onehot], dim=-1)) + self.gene_id_emb

        if self.use_latent_mask:
            if force_mask_kinases:
                m = self.kinase_like_mask.view(1, self.n_genes, 1)
                h0 = h0 * (1.0 - m) + self.mask_token * m
            elif self.training and self.mask_prob_train > 0.0:
                rand_m = (torch.rand(B, self.n_genes, 1, device=h0.device) < self.mask_prob_train).float()
                m = rand_m * self.kinase_like_mask.view(1, self.n_genes, 1)
                h0 = h0 * (1.0 - m) + self.mask_token * m

        m_pos = torch.matmul(self.A_pos, self.msg_pos(h0))
        m_neg = torch.matmul(self.A_neg, self.msg_neg(h0))
        h1 = self.layer1_norm(h0 + F.gelu(m_pos - m_neg))

        node_imp = torch.sigmoid(self.pw_attn_gate(h1))
        p_state = torch.matmul(self.H_g2p.t(), h1 * node_imp)
        pw_activity = p_state.mean(dim=-1)  # (B, P)
        pw_feat = self.pw_compress(pw_activity)  # (B, 32)

        gate = 1.0 + 0.5 * torch.tanh(self.target_gate(x_tstat))
        h_joint = torch.cat([self.ctrl_enc(x_ctrl), pw_feat, self.teach_enc(x_teach), x_tstat], dim=-1)
        delta_prog = self.prog_head(h_joint) @ self.U_prog
        return gate * (self.ridge_scale * x_ridge) + 0.5 * delta_prog, pw_activity

# 中文注释：本模块构建了基于全量人类先验通路的“基因-通路二部图”底座网络，
# 第1层沿 OmniPath 有向激活/抑制边做 1 跳传导，第2层并行聚合 3000+ 条人类通路状态并回传基因，
# 同时支持对缺失/未观测蛋白修饰的信号节点施加 [MASK] 隐式状态推断。
