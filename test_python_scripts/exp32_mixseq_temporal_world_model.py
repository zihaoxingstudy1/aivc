# -*- coding: utf-8 -*-
"""
exp32_mixseq_temporal_world_model.py
Exp32: 灰箱有向时序动力学世界模型 (MIXseq 跨 115 细胞系 0h -> 6h -> 24h)
严格按细胞系留出 20% 未见细胞系盲测：
  - 对比 Static Linear-Ridge vs Static MLP vs Exp32_GreyBox_OmniPath_ODE
  - 检验：
      1) 仅从 0h 基态预测 6h 与 24h 的跨细胞系泛化精度
      2) 给定早期 6h 实测状态，动力学推演 24h 终态的精度
      3) 导出药物处理后各生物通路从 0h -> 6h -> 12h -> 18h -> 24h 的真实级联激活顺序
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from pgnn import build_gene_pathway_bipartite_tensors

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5AD_PATH = r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad"
KG_NPZ = r"D:\data\AIVC\data\mixseq_drug_kg_features.npz"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp32_temporal_world_model_results.csv"
OUT_PATHWAY_CSV = r"D:\data\AIVC\test_results\exp32_temporal_pathway_cascade.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp32_temporal_cascade_plot.png"


def eval_delta_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    p_all, p_top20, d_top20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p_all.append(float(pearsonr(yt, yp)[0]))
        top_idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[top_idx]) > 1e-6 and np.std(yp[top_idx]) > 1e-6:
            p_top20.append(float(pearsonr(yt[top_idx], yp[top_idx])[0]))
        d_top20.append(float((np.sign(yt[top_idx]) == np.sign(yp[top_idx])).mean()))
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "model": label,
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": mse,
    }


class StaticTimeMLP(nn.Module):
    """静态基线：分别从 (ctrl_0h, drug_feat, time_scalar) 直接映射终点，无状态转移。"""

    def __init__(self, n_genes: int, d_drug: int, U_prog: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        k_prog = U_prog.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_genes + d_drug + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, k_prog),
        )

    def forward(self, x_ctrl, d_feat, t_norm):
        h = torch.cat([x_ctrl, d_feat, t_norm], dim=-1)
        return self.net(h) @ self.U_prog


class GreyBoxOmniPathTemporalODE(nn.Module):
    """
    灰箱有向时序动力学世界模型 (Exp32):
      dz/dt = -softplus(gamma) * z + tanh(W_cascade @ z) + u_drug(ctrl, drug) * exp(-k * t)
      其中 W_cascade 由 OmniPath 有向激活(+)/抑制(-) 先验矩阵在基因程序空间投影构成骨架。
    """

    def __init__(self, n_genes: int, d_drug: int, U_prog: np.ndarray, A_prog_prior: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))  # (K, G)
        self.register_buffer("A_prog_prior", torch.tensor(A_prog_prior, dtype=torch.float32))  # (K, K)
        k_prog = U_prog.shape[0]

        # 阻尼降解率 gamma > 0 与药物半衰衰减率 k_decay > 0
        self.gamma_raw = nn.Parameter(torch.zeros(k_prog))
        self.k_decay_raw = nn.Parameter(torch.tensor(-1.0))

        # 药物外力编码器 u_0(ctrl_0h, drug_feat) -> (B, K)
        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.25))
        self.drug_enc = nn.Sequential(nn.Linear(d_drug, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.u_head = nn.Sequential(
            nn.Linear(48 + 64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, k_prog),
        )

        # 受 OmniPath 先验引导的级联传播矩阵: alpha * A_prog_prior + 可学习低秩调控微调
        self.prior_scale = nn.Parameter(torch.tensor(0.5))
        self.W_low1 = nn.Parameter(torch.randn(k_prog, 12) * 0.03)
        self.W_low2 = nn.Parameter(torch.randn(12, k_prog) * 0.03)

    def _rhs(self, z: torch.Tensor, u0: torch.Tensor, t_val: float) -> torch.Tensor:
        gamma = F.softplus(self.gamma_raw) * 0.5
        k_dec = F.softplus(self.k_decay_raw)
        W_eff = self.prior_scale * self.A_prog_prior + (self.W_low1 @ self.W_low2)
        cascade = torch.tanh(z @ W_eff.t())
        force = u0 * torch.exp(-k_dec * t_val)
        return -gamma * z + 0.5 * cascade + force

    def integrate(self, z0: torch.Tensor, x_ctrl: torch.Tensor, d_feat: torch.Tensor, t_start: float, t_end: float, n_steps: int = 4) -> list:
        """从 t_start 积分到 t_end (单位: 6小时=0.25天，24小时=1.0天)，返回轨迹快照。"""
        u0 = self.u_head(torch.cat([self.ctrl_enc(x_ctrl), self.drug_enc(d_feat)], dim=-1))
        dt = (t_end - t_start) / float(n_steps)
        z = z0
        t_cur = t_start
        traj = [(t_cur, z @ self.U_prog)]
        for _ in range(n_steps):
            k1 = self._rhs(z, u0, t_cur)
            k2 = self._rhs(z + 0.5 * dt * k1, u0, t_cur + 0.5 * dt)
            k3 = self._rhs(z + 0.5 * dt * k2, u0, t_cur + 0.5 * dt)
            k4 = self._rhs(z + dt * k3, u0, t_cur + dt)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_cur += dt
            traj.append((t_cur, z @ self.U_prog))
        return traj


def main():
    print(f"=== Exp32: MIXseq 0h -> 6h -> 24h 灰箱有向时序世界模型 ({DEVICE}) ===")
    ad = sc.read_h5ad(H5AD_PATH)
    genes = list(ad.var_names)
    n_genes = len(genes)
    obs = ad.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)

    # 仅保留明确单时间点 6h 与 24h（不重复 log1p，MIXseq_hvg2000 已是 log1p 矩阵）
    sub_mask = obs["time"].isin(["6", "24"]).values
    ad_sub = ad[sub_mask].copy()
    so = ad_sub.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in so.columns:
        so[c] = so[c].astype(str)

    X_dense = ad_sub.X.toarray().astype(np.float32) if hasattr(ad_sub.X, "toarray") else np.asarray(ad_sub.X, dtype=np.float32)
    del ad, ad_sub

    ctrl_6, ctrl_24 = {}, {}
    for cl in so["cell_line"].unique():
        m6 = (so["cell_line"] == cl) & (so["time"] == "6") & (so["perturbation"] == "control")
        m24 = (so["cell_line"] == cl) & (so["time"] == "24") & (so["perturbation"] == "control")
        if m6.sum() >= 5:
            ctrl_6[cl] = X_dense[m6.values].mean(axis=0)
        if m24.sum() >= 5:
            ctrl_24[cl] = X_dense[m24.values].mean(axis=0)

    # 载入药物分子/靶点 KG 特征
    kg = np.load(KG_NPZ, allow_pickle=True)
    kg_keys = [str(x) for x in kg["drugs"]] if "drugs" in kg else list(kg.files)
    if "features" in kg:
        drug_feat_map = {str(d): kg["features"][i].astype(np.float32) for i, d in enumerate(kg["drugs"])}
    else:
        # 兼容 one-hot + 剂量编码兜底
        drug_feat_map = {}

    tr_so = so[so["perturbation"] != "control"]
    grp = tr_so.groupby(["cell_line", "perturbation", "dose_value", "time"]).indices
    pairs = sorted({(k[0], k[1], k[2]) for k in grp.keys()})

    all_drugs = sorted({p[1] for p in pairs})
    d2i = {d: i for i, d in enumerate(all_drugs)}

    records = []
    for cl, drug, dose in pairs:
        if cl not in ctrl_6 or cl not in ctrl_24:
            continue
        i6 = grp.get((cl, drug, dose, "6"))
        i24 = grp.get((cl, drug, dose, "24"))
        if i6 is None or i24 is None or len(i6) < 15 or len(i24) < 15:
            continue
        d6 = (X_dense[i6].mean(axis=0) - ctrl_6[cl]).astype(np.float32)
        d24 = (X_dense[i24].mean(axis=0) - ctrl_24[cl]).astype(np.float32)
        oh = np.zeros(len(all_drugs), dtype=np.float32)
        oh[d2i[drug]] = 1.0
        kg_v = drug_feat_map.get(drug, np.zeros(1, dtype=np.float32))
        dfeat = np.concatenate([oh, kg_v, np.array([np.log1p(float(dose))], dtype=np.float32)])
        records.append({
            "cell_line": cl,
            "drug": drug,
            "dose": dose,
            "ctrl": ctrl_6[cl].astype(np.float32),
            "dfeat": dfeat,
            "d6": d6,
            "d24": d24,
        })

    cls_all = sorted({r["cell_line"] for r in records})
    rng = np.random.RandomState(SEED)
    rng.shuffle(cls_all)
    n_test_cl = max(1, int(0.2 * len(cls_all)))
    test_cls = set(cls_all[:n_test_cl])
    tr_rec = [r for r in records if r["cell_line"] not in test_cls]
    ev_rec = [r for r in records if r["cell_line"] in test_cls]
    print(f"有效 0h->6h->24h 配对={len(records)} | 训练细胞系={len(cls_all)-n_test_cl} ({len(tr_rec)}条) | 盲测未见细胞系={n_test_cl} ({len(ev_rec)}条)")

    X_ctrl_tr = np.vstack([r["ctrl"] for r in tr_rec])
    X_df_tr = np.vstack([r["dfeat"] for r in tr_rec])
    Y6_tr = np.vstack([r["d6"] for r in tr_rec])
    Y24_tr = np.vstack([r["d24"] for r in tr_rec])

    X_ctrl_ev = np.vstack([r["ctrl"] for r in ev_rec])
    X_df_ev = np.vstack([r["dfeat"] for r in ev_rec])
    Y6_ev = np.vstack([r["d6"] for r in ev_rec])
    Y24_ev = np.vstack([r["d24"] for r in ev_rec])

    # 1. 基线 A: Static Linear-Ridge
    r6 = Ridge(alpha=20.0).fit(X_df_tr, Y6_tr)
    r24 = Ridge(alpha=20.0).fit(X_df_tr, Y24_tr)
    p6_lin = r6.predict(X_df_ev)
    p24_lin = r24.predict(X_df_ev)

    # 低秩程序空间 + OmniPath 有向先验投影
    svd = TruncatedSVD(n_components=48, random_state=SEED)
    svd.fit(np.vstack([Y6_tr, Y24_tr]))
    U_prog = svd.components_.astype(np.float32)  # (48, 2000)

    bipartite = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A_prog_prior = (U_prog @ A_signed @ U_prog.T).astype(np.float32)

    # 2. 基线 B: Static MLP
    torch.manual_seed(SEED)
    smlp = StaticTimeMLP(n_genes, X_df_tr.shape[1], U_prog).to(DEVICE)
    opt_s = torch.optim.AdamW(smlp.parameters(), lr=1.5e-3, weight_decay=1e-3)
    Xc_t = torch.tensor(X_ctrl_tr, device=DEVICE)
    Xd_t = torch.tensor(X_df_tr, device=DEVICE)
    Y6_t = torch.tensor(Y6_tr, device=DEVICE)
    Y24_t = torch.tensor(Y24_tr, device=DEVICE)
    t025 = torch.full((len(tr_rec), 1), 0.25, device=DEVICE)
    t100 = torch.full((len(tr_rec), 1), 1.00, device=DEVICE)
    smlp.train()
    for _ in range(120):
        opt_s.zero_grad()
        loss = F.mse_loss(smlp(Xc_t, Xd_t, t025), Y6_t) + F.mse_loss(smlp(Xc_t, Xd_t, t100), Y24_t)
        loss.backward()
        opt_s.step()
    smlp.eval()
    with torch.no_grad():
        Xc_e = torch.tensor(X_ctrl_ev, device=DEVICE)
        Xd_e = torch.tensor(X_df_ev, device=DEVICE)
        p6_mlp = smlp(Xc_e, Xd_e, torch.full((len(ev_rec), 1), 0.25, device=DEVICE)).cpu().numpy()
        p24_mlp = smlp(Xc_e, Xd_e, torch.full((len(ev_rec), 1), 1.00, device=DEVICE)).cpu().numpy()

    # 3. Exp32: GreyBox OmniPath Directed Temporal ODE (0h -> 6h -> 24h)
    torch.manual_seed(SEED)
    ode_net = GreyBoxOmniPathTemporalODE(n_genes, X_df_tr.shape[1], U_prog, A_prog_prior).to(DEVICE)
    opt_o = torch.optim.AdamW(ode_net.parameters(), lr=2e-3, weight_decay=1e-3)
    z0_tr = torch.zeros((len(tr_rec), 48), device=DEVICE)
    ode_net.train()
    for _ in range(150):
        opt_o.zero_grad()
        # 从 t=0 连续积分到 t=0.25(6h) 再到 t=1.0(24h)
        tr_6 = ode_net.integrate(z0_tr, Xc_t, Xd_t, t_start=0.0, t_end=0.25, n_steps=2)
        pred_6 = tr_6[-1][1]
        z_6 = pred_6 @ ode_net.U_prog.t()
        tr_24 = ode_net.integrate(z_6, Xc_t, Xd_t, t_start=0.25, t_end=1.0, n_steps=4)
        pred_24 = tr_24[-1][1]
        loss = F.mse_loss(pred_6, Y6_t) + F.mse_loss(pred_24, Y24_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_net.parameters(), 2.0)
        opt_o.step()

    ode_net.eval()
    with torch.no_grad():
        z0_ev = torch.zeros((len(ev_rec), 48), device=DEVICE)
        traj_full = ode_net.integrate(z0_ev, Xc_e, Xd_e, t_start=0.0, t_end=1.0, n_steps=8)
        # n_steps=8 -> t = [0, 0.125(3h), 0.25(6h), 0.375(9h), 0.5(12h), 0.625(15h), 0.75(18h), 0.875(21h), 1.0(24h)]
        p6_ode = traj_full[2][1].cpu().numpy()
        p24_ode = traj_full[-1][1].cpu().numpy()
        # Rollout from real observed 6h -> 24h
        z6_obs_ev = torch.tensor(Y6_ev @ U_prog.T, device=DEVICE)
        p24_rollout = ode_net.integrate(z6_obs_ev, Xc_e, Xd_e, t_start=0.25, t_end=1.0, n_steps=6)[-1][1].cpu().numpy()

    res_rows = [
        eval_delta_metrics(Y6_ev, p6_lin, "Static_Linear_6h"),
        eval_delta_metrics(Y6_ev, p6_mlp, "Static_MLP_6h"),
        eval_delta_metrics(Y6_ev, p6_ode, "Exp32_GreyBoxODE_0h->6h"),
        eval_delta_metrics(Y24_ev, p24_lin, "Static_Linear_24h"),
        eval_delta_metrics(Y24_ev, p24_mlp, "Static_MLP_24h"),
        eval_delta_metrics(Y24_ev, p24_ode, "Exp32_GreyBoxODE_0h->24h"),
        eval_delta_metrics(Y24_ev, p24_rollout, "Exp32_GreyBoxODE_Obs6h->24h"),
    ]
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Exp32 未见细胞系盲测对比 ({OUT_CSV}) ===")
    print(res_df.to_string(index=False))

    # 导出全轨迹关键生物通路时序激活曲线 (0h -> 24h)
    mem = pd.read_parquet(os.path.join(PRIORS_DIR, "gene_pathway_membership.parquet"))
    hm = mem[mem["source_db"] == "MSigDB_Hallmark_2020"]
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    focus_pws = [
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_P53_PATHWAY",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_GLYCOLYSIS",
    ]
    pw_rows = []
    for t_norm, pred_mat in traj_full:
        hrs = round(t_norm * 24.0, 1)
        arr = pred_mat.cpu().numpy()
        for pw in focus_pws:
            sub_g = hm[hm["pathway_name"] == pw]["gene_symbol"]
            idxs = [g2i[g.upper()] for g in sub_g if g.upper() in g2i]
            if idxs:
                pw_rows.append({"time_hours": hrs, "pathway": pw.replace("HALLMARK_", ""), "mean_abs_activity": float(np.mean(np.abs(arr[:, idxs]))), "signed_activity": float(np.mean(arr[:, idxs]))})
    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(OUT_PATHWAY_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    for pw, grp_df in pw_df.groupby("pathway"):
        ax.plot(grp_df["time_hours"], grp_df["signed_activity"], marker="o", linewidth=2.0, label=pw)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time Post-Perturbation (Hours)", fontsize=10)
    ax.set_ylabel("Predicted Pathway Activity (Signed Delta)", fontsize=10)
    ax.set_title("Exp32: Grey-Box OmniPath ODE Pathway Cascade (0h -> 24h, Unseen Cell Lines)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved pathway cascade CSV & plot -> {OUT_PATHWAY_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
