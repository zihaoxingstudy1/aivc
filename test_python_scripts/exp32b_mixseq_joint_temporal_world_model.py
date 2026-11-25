# -*- coding: utf-8 -*-
"""
exp32b_mixseq_joint_temporal_world_model.py
Exp32b: 联合全量 MIXseq (13种药 6h + 24h 共 ~1,100+ 跨171细胞系样本) + Obs6h->24h 转移一致性训练
严格留出 20% 未见细胞系盲测，对比 Static_Linear vs Static_MLP vs Exp32b_JointGreyBoxODE，
并导出 0h -> 6h -> 24h 关键 Hallmark 通路级联激活顺序图。
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
OUT_CSV = r"D:\data\AIVC\test_results\exp32b_temporal_world_model_results.csv"
OUT_PATHWAY_CSV = r"D:\data\AIVC\test_results\exp32b_temporal_pathway_cascade.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp32b_temporal_cascade_plot.png"


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
        return self.net(torch.cat([x_ctrl, d_feat, t_norm], dim=-1)) @ self.U_prog


class JointGreyBoxOmniPathODE(nn.Module):
    """
    联合全量数据的灰箱有向时序动力学模型 (Exp32b):
      - 支持同时在单时间点 (0->6h 或 0->24h) 与双时间点轨迹 (0->6->24h + Obs6h->24h) 上联合训练
      - 由 OmniPath 有向激活/抑制先验矩阵约束传播方向，带稳态阻尼与药物衰减项
    """

    def __init__(self, n_genes: int, d_drug: int, U_prog: np.ndarray, A_prog_prior: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.register_buffer("A_prog_prior", torch.tensor(A_prog_prior, dtype=torch.float32))
        k_prog = U_prog.shape[0]

        self.gamma_raw = nn.Parameter(torch.zeros(k_prog))
        self.k_decay_raw = nn.Parameter(torch.tensor(-0.5))

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25))
        self.drug_enc = nn.Sequential(nn.Linear(d_drug, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.u_head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, k_prog),
        )

        self.prior_scale = nn.Parameter(torch.tensor(0.5))
        self.W_low1 = nn.Parameter(torch.randn(k_prog, 16) * 0.03)
        self.W_low2 = nn.Parameter(torch.randn(16, k_prog) * 0.03)

    def _rhs(self, z: torch.Tensor, u0: torch.Tensor, t_val: float) -> torch.Tensor:
        gamma = F.softplus(self.gamma_raw) * 0.5
        k_dec = F.softplus(self.k_decay_raw)
        W_eff = self.prior_scale * self.A_prog_prior + (self.W_low1 @ self.W_low2)
        cascade = torch.tanh(z @ W_eff.t())
        force = u0 * torch.exp(-k_dec * t_val)
        return -gamma * z + 0.5 * cascade + force

    def integrate(self, z0: torch.Tensor, x_ctrl: torch.Tensor, d_feat: torch.Tensor, t_start: float, t_end: float, n_steps: int = 4) -> list:
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
    print(f"=== Exp32b: MIXseq 全量13药联合灰箱时序世界模型 ({DEVICE}) ===")
    ad = sc.read_h5ad(H5AD_PATH)
    genes = list(ad.var_names)
    n_genes = len(genes)
    obs = ad.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)

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

    kg = np.load(KG_NPZ, allow_pickle=True)
    drug_feat_map = {str(d): kg["features"][i].astype(np.float32) for i, d in enumerate(kg["drugs"])} if ("drugs" in kg and "features" in kg) else {}

    tr_so = so[so["perturbation"] != "control"]
    grp = tr_so.groupby(["cell_line", "perturbation", "dose_value", "time"]).indices
    all_drugs = sorted(tr_so["perturbation"].unique())
    d2i = {d: i for i, d in enumerate(all_drugs)}

    def make_dfeat(drug, dose):
        oh = np.zeros(len(all_drugs), dtype=np.float32)
        oh[d2i[drug]] = 1.0
        kg_v = drug_feat_map.get(drug, np.zeros(1, dtype=np.float32))
        return np.concatenate([oh, kg_v, np.array([np.log1p(float(dose))], dtype=np.float32)])

    # 1. 收集所有单时间点样本 (6h 或 24h) 与双时间点配对样本
    single_6, single_24, paired = [], [], []
    keys_3 = sorted({(k[0], k[1], k[2]) for k in grp.keys()})
    for cl, drug, dose in keys_3:
        i6 = grp.get((cl, drug, dose, "6"))
        i24 = grp.get((cl, drug, dose, "24"))
        dfeat = make_dfeat(drug, dose)
        if i6 is not None and len(i6) >= 15 and cl in ctrl_6:
            d6 = (X_dense[i6].mean(axis=0) - ctrl_6[cl]).astype(np.float32)
            single_6.append({"cell_line": cl, "drug": drug, "ctrl": ctrl_6[cl], "dfeat": dfeat, "y": d6})
        if i24 is not None and len(i24) >= 15 and cl in ctrl_24:
            d24 = (X_dense[i24].mean(axis=0) - ctrl_24[cl]).astype(np.float32)
            single_24.append({"cell_line": cl, "drug": drug, "ctrl": ctrl_24[cl], "dfeat": dfeat, "y": d24})
        if i6 is not None and i24 is not None and len(i6) >= 15 and len(i24) >= 15 and cl in ctrl_6 and cl in ctrl_24:
            d6 = (X_dense[i6].mean(axis=0) - ctrl_6[cl]).astype(np.float32)
            d24 = (X_dense[i24].mean(axis=0) - ctrl_24[cl]).astype(np.float32)
            paired.append({"cell_line": cl, "drug": drug, "ctrl": ctrl_6[cl], "dfeat": dfeat, "d6": d6, "d24": d24})

    # 与 Exp32 完全相同的 22 个盲测细胞系留出集合
    paired_cls = sorted({r["cell_line"] for r in paired})
    rng = np.random.RandomState(SEED)
    rng.shuffle(paired_cls)
    test_cls = set(paired_cls[: max(1, int(0.2 * len(paired_cls)))])

    tr_6 = [r for r in single_6 if r["cell_line"] not in test_cls]
    tr_24 = [r for r in single_24 if r["cell_line"] not in test_cls]
    tr_pair = [r for r in paired if r["cell_line"] not in test_cls]
    ev_pair = [r for r in paired if r["cell_line"] in test_cls]
    ev_24_all = [r for r in single_24 if r["cell_line"] in test_cls]

    print(f"全量数据: 6h训练={len(tr_6)}, 24h训练={len(tr_24)}, 双时间配对训练={len(tr_pair)} | 盲测未见22系配对={len(ev_pair)}, 全13药24h盲测={len(ev_24_all)}")

    svd = TruncatedSVD(n_components=48, random_state=SEED)
    svd.fit(np.vstack([r["y"] for r in tr_6 + tr_24]))
    U_prog = svd.components_.astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A_prog_prior = (U_prog @ A_signed @ U_prog.T).astype(np.float32)

    # 准备 GPU 张量
    def to_gpu(lst, y_key="y"):
        return (
            torch.tensor(np.vstack([r["ctrl"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r["dfeat"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r[y_key] for r in lst]), device=DEVICE),
        )

    Xc6, Xd6, Y6 = to_gpu(tr_6)
    Xc24, Xd24, Y24 = to_gpu(tr_24)
    Xcp, Xdp, Y6p = to_gpu(tr_pair, "d6")
    _, _, Y24p = to_gpu(tr_pair, "d24")

    Xc_ev = torch.tensor(np.vstack([r["ctrl"] for r in ev_pair]), device=DEVICE)
    Xd_ev = torch.tensor(np.vstack([r["dfeat"] for r in ev_pair]), device=DEVICE)
    Y6_ev = np.vstack([r["d6"] for r in ev_pair])
    Y24_ev = np.vstack([r["d24"] for r in ev_pair])

    Xc_ev24 = torch.tensor(np.vstack([r["ctrl"] for r in ev_24_all]), device=DEVICE)
    Xd_ev24 = torch.tensor(np.vstack([r["dfeat"] for r in ev_24_all]), device=DEVICE)
    Y24_ev_all = np.vstack([r["y"] for r in ev_24_all])

    # 基线 1: Ridge
    r6 = Ridge(alpha=20.0).fit(Xd6.cpu().numpy(), Y6.cpu().numpy())
    r24 = Ridge(alpha=20.0).fit(Xd24.cpu().numpy(), Y24.cpu().numpy())

    # 基线 2: 全量联合 Static MLP
    torch.manual_seed(SEED)
    smlp = StaticTimeMLP(n_genes, Xd6.shape[1], U_prog).to(DEVICE)
    opt_s = torch.optim.AdamW(smlp.parameters(), lr=1.5e-3, weight_decay=1e-3)
    t025_tr = torch.full((len(tr_6), 1), 0.25, device=DEVICE)
    t100_tr = torch.full((len(tr_24), 1), 1.00, device=DEVICE)
    smlp.train()
    for _ in range(140):
        opt_s.zero_grad()
        loss = F.mse_loss(smlp(Xc6, Xd6, t025_tr), Y6) + F.mse_loss(smlp(Xc24, Xd24, t100_tr), Y24)
        loss.backward()
        opt_s.step()
    smlp.eval()
    with torch.no_grad():
        p6_mlp = smlp(Xc_ev, Xd_ev, torch.full((len(ev_pair), 1), 0.25, device=DEVICE)).cpu().numpy()
        p24_mlp = smlp(Xc_ev, Xd_ev, torch.full((len(ev_pair), 1), 1.00, device=DEVICE)).cpu().numpy()
        p24_mlp_all = smlp(Xc_ev24, Xd_ev24, torch.full((len(ev_24_all), 1), 1.00, device=DEVICE)).cpu().numpy()

    # Exp32b: 全量13药联合灰箱 ODE + Obs6h->24h 转移一致性监督
    torch.manual_seed(SEED)
    ode = JointGreyBoxOmniPathODE(n_genes, Xd6.shape[1], U_prog, A_prog_prior).to(DEVICE)
    opt_o = torch.optim.AdamW(ode.parameters(), lr=2e-3, weight_decay=1e-3)
    z0_6 = torch.zeros((len(tr_6), 48), device=DEVICE)
    z0_24 = torch.zeros((len(tr_24), 48), device=DEVICE)
    z6p_obs = Y6p @ ode.U_prog.t()

    ode.train()
    for _ in range(160):
        opt_o.zero_grad()
        p6_tr = ode.integrate(z0_6, Xc6, Xd6, t_start=0.0, t_end=0.25, n_steps=2)[-1][1]
        p24_tr = ode.integrate(z0_24, Xc24, Xd24, t_start=0.0, t_end=1.0, n_steps=6)[-1][1]
        # 显式加上带轻量噪声的 Obs6h -> 24h 状态转移训练
        z6_noisy = z6p_obs + 0.05 * torch.randn_like(z6p_obs)
        p24_roll_tr = ode.integrate(z6_noisy, Xcp, Xdp, t_start=0.25, t_end=1.0, n_steps=4)[-1][1]

        loss = F.mse_loss(p6_tr, Y6) + F.mse_loss(p24_tr, Y24) + 0.5 * F.mse_loss(p24_roll_tr, Y24p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode.parameters(), 2.0)
        opt_o.step()

    ode.eval()
    with torch.no_grad():
        z0_ev = torch.zeros((len(ev_pair), 48), device=DEVICE)
        traj_full = ode.integrate(z0_ev, Xc_ev, Xd_ev, t_start=0.0, t_end=1.0, n_steps=8)
        p6_ode = traj_full[2][1].cpu().numpy()
        p24_ode = traj_full[-1][1].cpu().numpy()
        z6_obs_ev = torch.tensor(Y6_ev @ U_prog.T, device=DEVICE)
        p24_rollout = ode.integrate(z6_obs_ev, Xc_ev, Xd_ev, t_start=0.25, t_end=1.0, n_steps=6)[-1][1].cpu().numpy()
        p24_ode_all = ode.integrate(torch.zeros((len(ev_24_all), 48), device=DEVICE), Xc_ev24, Xd_ev24, t_start=0.0, t_end=1.0, n_steps=6)[-1][1].cpu().numpy()

    res_rows = [
        eval_delta_metrics(Y6_ev, r6.predict(Xd_ev.cpu().numpy()), "Static_Linear_6h(pair)"),
        eval_delta_metrics(Y6_ev, p6_mlp, "Static_MLP_6h(pair)"),
        eval_delta_metrics(Y6_ev, p6_ode, "Exp32b_JointGreyBoxODE_0h->6h(pair)"),
        eval_delta_metrics(Y24_ev, r24.predict(Xd_ev.cpu().numpy()), "Static_Linear_24h(pair)"),
        eval_delta_metrics(Y24_ev, p24_mlp, "Static_MLP_24h(pair)"),
        eval_delta_metrics(Y24_ev, p24_ode, "Exp32b_JointGreyBoxODE_0h->24h(pair)"),
        eval_delta_metrics(Y24_ev, p24_rollout, "Exp32b_JointGreyBoxODE_Obs6h->24h(pair)"),
        eval_delta_metrics(Y24_ev_all, r24.predict(Xd_ev24.cpu().numpy()), "Static_Linear_24h(All13Drugs)"),
        eval_delta_metrics(Y24_ev_all, p24_mlp_all, "Static_MLP_24h(All13Drugs)"),
        eval_delta_metrics(Y24_ev_all, p24_ode_all, "Exp32b_JointGreyBoxODE_0h->24h(All13Drugs)"),
    ]
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Exp32b 未见22细胞系盲测对比 ({OUT_CSV}) ===")
    print(res_df.to_string(index=False))

    # 导出并绘制 0h -> 24h Hallmark 通路级联激活顺序图 (以全13药 24h 与配对药物轨迹为例)
    mem = pd.read_parquet(os.path.join(PRIORS_DIR, "gene_pathway_membership.parquet"))
    hm = mem[mem["source_db"] == "MSigDB_Hallmark_2020"].copy()
    hm["pw_low"] = hm["pathway_name"].str.lower()
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    focus_keywords = ["tnf-alpha", "p53 pathway", "e2f targets", "g2-m checkpoint", "apoptosis", "glycolysis"]

    pw_rows = []
    for t_norm, pred_mat in traj_full:
        hrs = round(t_norm * 24.0, 1)
        arr = pred_mat.cpu().numpy()
        for kw in focus_keywords:
            matched = hm[hm["pw_low"].str.contains(kw, regex=False)]
            if len(matched) == 0:
                continue
            pw_name = str(matched["pathway_name"].iloc[0])
            idxs = [g2i[g.upper()] for g in matched["gene_symbol"] if g.upper() in g2i]
            if idxs:
                pw_rows.append({
                    "time_hours": hrs,
                    "pathway": pw_name,
                    "mean_abs_activity": float(np.mean(np.abs(arr[:, idxs]))),
                    "signed_activity": float(np.mean(arr[:, idxs])),
                })

    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(OUT_PATHWAY_CSV, index=False, encoding="utf-8-sig")
    if len(pw_df) > 0:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
        for pw, grp_df in pw_df.groupby("pathway"):
            ax.plot(grp_df["time_hours"], grp_df["signed_activity"], marker="o", linewidth=2.0, label=pw)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time Post-Perturbation (Hours)", fontsize=10)
        ax.set_ylabel("Predicted Pathway Activity (Signed Delta)", fontsize=10)
        ax.set_title("Exp32b: Joint Grey-Box OmniPath ODE Pathway Cascade (0h -> 24h, Unseen Cell Lines)", fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8.5, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=180)
        print(f"Saved pathway cascade CSV & plot -> {OUT_PATHWAY_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
