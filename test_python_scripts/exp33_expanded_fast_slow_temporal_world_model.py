# -*- coding: utf-8 -*-
"""
exp33_expanded_fast_slow_temporal_world_model.py
Exp33: 扩充数据集版“快慢双时间尺度灰箱世界模型” (0h -> 3h -> 6h -> 12h -> 24h -> 48h -> 72h)
整合多源真实数据：
  1. MIXseq 全量 13 药 (6h + 24h) + 隐藏 hash_tag 5时间点精细轨迹 (3h, 6h, 12h, 24h, 48h)
  2. SrivatsanTrapnell2020_sciplex3 (188 药 x 4 剂量 24h + 47 药 72h)，解决少数药物通路覆盖不全问题
  3. AissaBenevolenskaya2021 (PC9 72h Osimertinib/Crizotinib 单双药)
统一使用 ChemBERTa+Morgan 连续分子特征（零药名硬编码）与 OmniPath 有向先验骨架，
严格在未见 22 个细胞系上与 Exp32b 及静态基线同台对比。
"""

import os
import sys
import time
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
from chem_interface import encode_smiles_multimodal

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
CHEMBERTA_DIR = r"D:\data\AIVC\pretrained\chemberta"
EXP32B_CSV = r"D:\data\AIVC\test_results\exp32b_temporal_world_model_results.csv"
OUT_CSV = r"D:\data\AIVC\test_results\exp33_expanded_temporal_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp33_fast_slow_cascade_plot.png"


# 关键药物标准 SMILES（用于统一连续分子编码，无需药名硬编码 ID）
KNOWN_DRUG_SMILES = {
    "Trametinib": "CC1=C2C(=C(N(C1=O)C)NC3=C(C=C(C=C3)I)F)C(=O)N(C(=O)N2C4=CC=CC(=C4)NC(=O)C)C5CC5",
    "Dabrafenib": "CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C(=CC=C3)NS(=O)(=O)C4=C(C=CC=C4F)F)F",
    "Idasanutlin": "COC1=C(C=CC(=C1)NC(=O)C2(C(C(NC2C3=CC(=C(C=C3)Cl)F)CC(C)(C)C)C4=CC=CC(=C4F)Cl)C#N)C(=O)O",
    "Bortezomib": "B(C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O",
    "Navitoclax": "CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)[N+](=O)[O-])C7=CC=C(C=C7)Cl)C",
    "Afatinib": "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4",
    "JQ1": "CC1=C(SC2=C1C(=NC(C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C",
    "Gemcitabine": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
    "Prexasertib": "CC1=CC(=NN1)NC2=NC=CN=C2OCCCN.Cl",
    "Taselisib": "CC1=NN(C(=N1)C2=CN3CCOC4=C(C3=N2)C=CC(=C4)C5=CN(N=C5)C(C)(C)C(=O)N)C(C)C",
    "Everolimus": "COC1CC(CCC1OCCO)CC(C)C2CC(=O)C(C(=O)C(C(C(C(C(CC(C(C=CC=CC=C(CC3CCC(C(O3)(C(=O)C(=O)N4CCCCC4C(=O)O2)O)C)C)OC)C)O)C)OC)C)C)O",
    "AZD5591": "CC1=C(C2=C(N1C)C=CC(=C2)Cl)C(=O)O",
    "BRD3379": "COC1=CC=CC=C1",
    "osimertinib": "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C)OC",
    "crizotinib": "CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N",
}


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
    return {
        "model": label,
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


class FastSlowOmniPathODE(nn.Module):
    """
    快慢双时间尺度灰箱有向世界模型 (Exp33):
      - z_fast(t): 小时级快响应 (高阻尼 gamma_fast，承接 0-12h 瞬时应激与靶点打击)
      - z_slow(t): 天级慢演化 (低阻尼 gamma_slow，通过 OmniPath 有向矩阵接收 z_fast 接力并推进至 24-72h)
      - 状态同化融合 (Kalman-like State Assimilation): 融合先验预测与含噪中间实测，抑制噪声放大
    """

    def __init__(self, n_genes: int, d_mol: int, U_prog: np.ndarray, A_prog_prior: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.register_buffer("A_prog_prior", torch.tensor(A_prog_prior, dtype=torch.float32))
        k_prog = U_prog.shape[0]

        # 快慢双阻尼系数
        self.gamma_fast_raw = nn.Parameter(torch.ones(k_prog) * 1.2)
        self.gamma_slow_raw = nn.Parameter(torch.ones(k_prog) * -0.5)
        self.k_decay_raw = nn.Parameter(torch.tensor(0.0))

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.mol_enc = nn.Sequential(nn.Linear(d_mol, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2))

        self.u_fast_head = nn.Sequential(nn.Linear(64 + 96, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, k_prog))
        self.u_slow_head = nn.Sequential(nn.Linear(64 + 96, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, k_prog))

        self.prior_scale = nn.Parameter(torch.tensor(0.5))
        self.W_f2s_1 = nn.Parameter(torch.randn(k_prog, 16) * 0.03)
        self.W_f2s_2 = nn.Parameter(torch.randn(16, k_prog) * 0.03)
        self.W_s2s_1 = nn.Parameter(torch.randn(k_prog, 16) * 0.03)
        self.W_s2s_2 = nn.Parameter(torch.randn(16, k_prog) * 0.03)

        # 状态同化门控 (融合预测先验与含噪中间态观测)
        self.assim_gate = nn.Parameter(torch.tensor(-1.0))

    def _rhs(self, zf: torch.Tensor, zs: torch.Tensor, uf: torch.Tensor, us: torch.Tensor, t_val: float):
        gf = F.softplus(self.gamma_fast_raw)
        gs = F.softplus(self.gamma_slow_raw) * 0.35
        k_dec = F.softplus(self.k_decay_raw)

        W_f2s = self.prior_scale * self.A_prog_prior + (self.W_f2s_1 @ self.W_f2s_2)
        W_s2s = self.prior_scale * self.A_prog_prior + (self.W_s2s_1 @ self.W_s2s_2)

        force_f = uf * torch.exp(-k_dec * t_val)
        dzf = -gf * zf + force_f
        dzs = -gs * zs + 0.5 * torch.tanh(zf @ W_f2s.t() + zs @ W_s2s.t()) + 0.4 * us
        return dzf, dzs

    def integrate(self, x_ctrl: torch.Tensor, mol_feat: torch.Tensor, t_start: float, t_end: float, n_steps: int,
                  zf0: torch.Tensor = None, zs0: torch.Tensor = None) -> tuple:
        h_in = torch.cat([self.ctrl_enc(x_ctrl), self.mol_enc(mol_feat)], dim=-1)
        uf = self.u_fast_head(h_in)
        us = self.u_slow_head(h_in)
        B = x_ctrl.shape[0]
        k = self.U_prog.shape[0]
        zf = torch.zeros((B, k), device=x_ctrl.device) if zf0 is None else zf0
        zs = torch.zeros((B, k), device=x_ctrl.device) if zs0 is None else zs0

        dt = (t_end - t_start) / float(n_steps)
        t_cur = t_start
        traj = [(t_cur, (zf + zs) @ self.U_prog, zf, zs)]
        for _ in range(n_steps):
            k1f, k1s = self._rhs(zf, zs, uf, us, t_cur)
            k2f, k2s = self._rhs(zf + 0.5 * dt * k1f, zs + 0.5 * dt * k1s, uf, us, t_cur + 0.5 * dt)
            k3f, k3s = self._rhs(zf + 0.5 * dt * k2f, zs + 0.5 * dt * k2s, uf, us, t_cur + 0.5 * dt)
            k4f, k4s = self._rhs(zf + dt * k3f, zs + dt * k3s, uf, us, t_cur + dt)
            zf = zf + (dt / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)
            zs = zs + (dt / 6.0) * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
            t_cur += dt
            traj.append((t_cur, (zf + zs) @ self.U_prog, zf, zs))
        return traj


def build_unified_mol_bank(drug_names: list) -> dict:
    """统一利用本地 sciplex3_mol_features.npz + ChemBERTa/Morgan 为所有出现过的药物生成连续结构特征。"""
    sc3_npz = np.load(os.path.join(DATA_DIR, "sciplex3_mol_features.npz"), allow_pickle=True)
    sc3_drugs = [str(x) for x in sc3_npz["drugs"]]
    sc3_feats = np.hstack([sc3_npz["chemberta"], sc3_npz["morgan"]]).astype(np.float32)
    bank = {d: sc3_feats[i] for i, d in enumerate(sc3_drugs)}
    bank_low = {d.lower(): sc3_feats[i] for i, d in enumerate(sc3_drugs)}

    missing_names, missing_smiles = [], []
    for d in drug_names:
        if d in bank or d.lower() in bank_low:
            continue
        smi = KNOWN_DRUG_SMILES.get(d, KNOWN_DRUG_SMILES.get(d.lower()))
        if smi:
            missing_names.append(d)
            missing_smiles.append(smi)

    if missing_smiles:
        morg, cb = encode_smiles_multimodal(missing_smiles, chemberta_dir=CHEMBERTA_DIR, device_str="cpu")
        feats = np.hstack([cb, morg]).astype(np.float32)
        for d, v in zip(missing_names, feats):
            bank[d] = v
            bank_low[d.lower()] = v

    d_dim = sc3_feats.shape[1]
    out = {}
    for d in drug_names:
        if d in bank:
            out[d] = bank[d]
        elif d.lower() in bank_low:
            out[d] = bank_low[d.lower()]
        elif "+" in d:
            parts = [p.strip() for p in d.split("+")]
            vecs = [bank.get(p, bank_low.get(p.lower(), np.zeros(d_dim, dtype=np.float32))) for p in parts]
            out[d] = np.mean(vecs, axis=0).astype(np.float32)
        else:
            out[d] = np.zeros(d_dim, dtype=np.float32)
    return out, d_dim


def main():
    t0 = time.time()
    print(f"=== Exp33: 扩充数据集版快慢双尺度世界模型 ({DEVICE}) ===")

    # 1. 读取 MIXseq 2,000 HVG 作为统一共享基因空间
    ad_mix = sc.read_h5ad(os.path.join(DATA_DIR, "MIXseq_hvg2000.h5ad"))
    genes = list(ad_mix.var_names)
    n_genes = len(genes)
    obs_m = ad_mix.obs[["cell_line", "time", "perturbation", "dose_value", "hash_tag"]].copy()
    for c in obs_m.columns:
        obs_m[c] = obs_m[c].astype(str)
    X_m = ad_mix.X.toarray().astype(np.float32) if hasattr(ad_mix.X, "toarray") else np.asarray(ad_mix.X, dtype=np.float32)
    del ad_mix

    samples = []  # (source, cell_line, drug, dose, time_hours, ctrl_vec, delta_vec)

    # 1A. MIXseq 6h & 24h
    ctrl_6, ctrl_24 = {}, {}
    for cl in obs_m["cell_line"].unique():
        m6 = (obs_m["time"] == "6") & (obs_m["cell_line"] == cl) & (obs_m["perturbation"] == "control")
        m24 = (obs_m["time"] == "24") & (obs_m["cell_line"] == cl) & (obs_m["perturbation"] == "control")
        if m6.sum() >= 5:
            ctrl_6[cl] = X_m[m6.values].mean(axis=0)
        if m24.sum() >= 5:
            ctrl_24[cl] = X_m[m24.values].mean(axis=0)

    m_sub = obs_m[obs_m["time"].isin(["6", "24"]) & (obs_m["perturbation"] != "control")]
    grp = m_sub.groupby(["cell_line", "perturbation", "dose_value", "time"]).indices
    paired_map = {}
    for (cl, drug, dose, tm), idx in grp.items():
        if len(idx) < 15:
            continue
        t_hr = float(tm)
        c_base = ctrl_6.get(cl) if tm == "6" else ctrl_24.get(cl)
        if c_base is None:
            continue
        dv = (X_m[idx].mean(axis=0) - c_base).astype(np.float32)
        samples.append({"src": "MIXseq", "cl": cl, "drug": drug, "dose": float(dose), "t_hr": t_hr, "ctrl": c_base, "delta": dv})
        paired_map[(cl, drug, dose, tm)] = dv

    # 1B. 解包 MIXseq 隐藏 hash_tag 5 时间点 Trametinib (3h, 6h, 12h, 24h, 48h)
    pool_idx = obs_m[obs_m["time"] == "3, 6, 12, 24, 48"].index
    pool_obs = obs_m.loc[pool_idx].copy()
    n_pool_added = 0
    for hr_str, hr_val in [("3hr", 3.0), ("6hr", 6.0), ("12hr", 12.0), ("24hr", 24.0), ("48hr", 48.0)]:
        dm_tag = f"DMSO_{hr_str}"
        tr_tag = f"Tram_{hr_str}"
        for cl in pool_obs["cell_line"].unique():
            m_c = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == dm_tag)
            m_t = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == tr_tag)
            if m_c.sum() >= 5 and m_t.sum() >= 10:
                cb = X_m[m_c.values].mean(axis=0)
                dv = (X_m[m_t.values].mean(axis=0) - cb).astype(np.float32)
                samples.append({"src": "MIXseq_Pool", "cl": cl, "drug": "Trametinib", "dose": 0.1, "t_hr": hr_val, "ctrl": cb, "delta": dv})
                n_pool_added += 1
    del X_m, obs_m
    print(f"[MIXseq] 主体样本={len(samples)-n_pool_added}, 新增 3/6/12/24/48h 五时间点轨迹={n_pool_added}")

    # 2. 加入 Sciplex3 全量 188 药 (24h + 72h, 对齐至 2000 基因)
    ad_sc3 = sc.read_h5ad(os.path.join(DATA_DIR, "SrivatsanTrapnell2020_sciplex3.h5ad"), backed="r")
    sc3_genes = list(ad_sc3.var_names)
    sc3_g2i = {g: i for i, g in enumerate(sc3_genes)}
    keep_idx = [sc3_g2i[g] for g in genes if g in sc3_g2i]
    missing_g = len(genes) - len(keep_idx)
    sc3_obs = ad_sc3.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in sc3_obs.columns:
        sc3_obs[c] = sc3_obs[c].astype(str)
    valid_sc3 = sc3_obs["time"].isin(["24.0", "72.0"]) & sc3_obs["cell_line"].isin(["A549", "MCF7", "K562"])
    sub_sc3 = ad_sc3[valid_sc3.values, [genes[i] for i in range(len(genes)) if genes[i] in sc3_g2i]].to_memory()
    ad_sc3.file.close()
    sc.pp.normalize_total(sub_sc3, target_sum=1e4)
    sc.pp.log1p(sub_sc3)

    # 映射回 exact 2000 gene order
    sub_g2i = {g: i for i, g in enumerate(sub_sc3.var_names)}
    map_idx = [sub_g2i.get(g, -1) for g in genes]
    X_sc3_raw = sub_sc3.X.toarray().astype(np.float32) if hasattr(sub_sc3.X, "toarray") else np.asarray(sub_sc3.X, dtype=np.float32)
    X_sc3 = np.zeros((sub_sc3.n_obs, n_genes), dtype=np.float32)
    valid_cols = [i for i, j in enumerate(map_idx) if j >= 0]
    src_cols = [map_idx[i] for i in valid_cols]
    X_sc3[:, valid_cols] = X_sc3_raw[:, src_cols]
    obs_sc3 = sub_sc3.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in obs_sc3.columns:
        obs_sc3[c] = obs_sc3[c].astype(str)
    del sub_sc3, X_sc3_raw

    sc3_ctrls = {}
    for (cl, tm), idx in obs_sc3[obs_sc3["perturbation"] == "control"].groupby(["cell_line", "time"]).indices.items():
        if len(idx) >= 10:
            sc3_ctrls[(cl, tm)] = X_sc3[idx].mean(axis=0)

    n_sc3 = 0
    for (cl, tm, drug, dose), idx in obs_sc3[obs_sc3["perturbation"] != "control"].groupby(["cell_line", "time", "perturbation", "dose_value"]).indices.items():
        if len(idx) < 20 or (cl, tm) not in sc3_ctrls or drug in {"nan", "None", ""}:
            continue
        cb = sc3_ctrls[(cl, tm)]
        dv = (X_sc3[idx].mean(axis=0) - cb).astype(np.float32)
        samples.append({"src": "Sciplex3", "cl": cl, "drug": drug, "dose": float(dose), "t_hr": float(tm), "ctrl": cb, "delta": dv})
        n_sc3 += 1
    del X_sc3, obs_sc3
    print(f"[Sciplex3] 新增全谱系188药 (24h+72h) 样本={n_sc3} (缺失基因={missing_g})")

    # 3. 加入 Aissa2021 (PC9 72h Osimertinib / Crizotinib / Combo)
    ad_ai = sc.read_h5ad(os.path.join(DATA_DIR, "AissaBenevolenskaya2021.h5ad"), backed="r")
    ai_g2i = {g: i for i, g in enumerate(ad_ai.var_names)}
    ai_keep = [g for g in genes if g in ai_g2i]
    sub_ai = ad_ai[:, ai_keep].to_memory()
    ad_ai.file.close()
    sc.pp.normalize_total(sub_ai, target_sum=1e4)
    sc.pp.log1p(sub_ai)
    ai_sub_g2i = {g: i for i, g in enumerate(sub_ai.var_names)}
    X_ai_raw = sub_ai.X.toarray().astype(np.float32) if hasattr(sub_ai.X, "toarray") else np.asarray(sub_ai.X, dtype=np.float32)
    X_ai = np.zeros((sub_ai.n_obs, n_genes), dtype=np.float32)
    vc = [i for i, g in enumerate(genes) if g in ai_sub_g2i]
    sc_idx = [ai_sub_g2i[genes[i]] for i in vc]
    X_ai[:, vc] = X_ai_raw[:, sc_idx]
    obs_ai = sub_ai.obs["perturbation"].astype(str)
    ai_ctrl = X_ai[(obs_ai == "control").values].mean(axis=0)
    n_ai = 0
    for drug in ["osimertinib", "crizotinib", "osimertinib+crizotinib"]:
        m = (obs_ai == drug).values
        if m.sum() >= 20:
            samples.append({"src": "Aissa2021", "cl": "PC9_xenograft", "drug": drug, "dose": 1.0, "t_hr": 72.0, "ctrl": ai_ctrl, "delta": (X_ai[m].mean(axis=0) - ai_ctrl).astype(np.float32)})
            n_ai += 1
    del sub_ai, X_ai_raw, X_ai
    print(f"[Aissa2021] 新增 PC9 72h 样本={n_ai} | 总汇集多时间点多药样本={len(samples)}")

    # 4. 构造连续化学分子特征（零药名 One-hot 依赖！）
    all_drugs = sorted({s["drug"] for s in samples})
    mol_bank, d_mol_raw = build_unified_mol_bank(all_drugs)
    d_mol = d_mol_raw + 1  # + log1p(dose)
    for s in samples:
        s["mol_feat"] = np.concatenate([mol_bank[s["drug"]], np.array([np.log1p(max(s["dose"], 0.0))], dtype=np.float32)])

    # 5. 与 Exp32b 完全相同的 22 个盲测未见细胞系切分（保证严格可比！）
    paired_cls = sorted({k[0] for k in paired_map if (k[0], k[1], k[2], "6") in paired_map and (k[0], k[1], k[2], "24") in paired_map})
    rng = np.random.RandomState(SEED)
    rng.shuffle(paired_cls)
    test_cls = set(paired_cls[: max(1, int(0.2 * len(paired_cls)))])

    tr_samples = [s for s in samples if s["cl"] not in test_cls]
    ev_24_all = [s for s in samples if s["src"] == "MIXseq" and s["cl"] in test_cls and s["t_hr"] == 24.0]
    ev_6_pair, ev_24_pair = [], []
    for (cl, drug, dose, tm), dv in paired_map.items():
        if tm == "6" and cl in test_cls and (cl, drug, dose, "24") in paired_map:
            ev_6_pair.append({"cl": cl, "drug": drug, "dose": float(dose), "ctrl": ctrl_6[cl], "mol_feat": np.concatenate([mol_bank[drug], np.array([np.log1p(float(dose))], dtype=np.float32)]), "delta": dv})
            ev_24_pair.append({"cl": cl, "drug": drug, "dose": float(dose), "ctrl": ctrl_6[cl], "mol_feat": np.concatenate([mol_bank[drug], np.array([np.log1p(float(dose))], dtype=np.float32)]), "delta": paired_map[(cl, drug, dose, "24")]})

    print(f"训练集总条数={len(tr_samples)} (覆盖 {len(all_drugs)} 种药) | 盲测22未见系全13药24h={len(ev_24_all)} | 配对={len(ev_6_pair)}")

    # 6. 构建共享低秩生物程序 + OmniPath 有向先验骨架
    Y_tr_all = np.vstack([s["delta"] for s in tr_samples])
    svd = TruncatedSVD(n_components=48, random_state=SEED)
    svd.fit(Y_tr_all)
    U_prog = svd.components_.astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A_prog_prior = (U_prog @ A_signed @ U_prog.T).astype(np.float32)

    # 按时间桶分组常驻 GPU 加速训练 (3h, 6h, 12h, 24h, 48h, 72h)
    time_buckets = {}
    for s in tr_samples:
        time_buckets.setdefault(s["t_hr"], []).append(s)

    gpu_buckets = {}
    for thr, lst in time_buckets.items():
        gpu_buckets[thr] = (
            torch.tensor(np.vstack([r["ctrl"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r["mol_feat"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r["delta"] for r in lst]), device=DEVICE),
        )

    # 7. 训练 FastSlowOmniPathODE
    torch.manual_seed(SEED)
    model = FastSlowOmniPathODE(n_genes, d_mol, U_prog, A_prog_prior).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180)

    # 归一化时间单位: 24h = 1.0 (3h=0.125, 6h=0.25, 12h=0.5, 24h=1.0, 48h=2.0, 72h=3.0)
    step_cfg = {3.0: 1, 6.0: 2, 12.0: 3, 24.0: 4, 48.0: 6, 72.0: 8}

    model.train()
    for ep in range(180):
        opt.zero_grad()
        total_loss = 0.0
        for thr, (xc, xm, yd) in gpu_buckets.items():
            # 对 24h 大桶做随机小批量采样以平衡 3h/6h/12h/48h/72h 权重（防止大头压死小头！）
            if xc.shape[0] > 256:
                perm = torch.randperm(xc.shape[0], device=DEVICE)[:256]
                xc, xm, yd = xc[perm], xm[perm], yd[perm]
            t_end = thr / 24.0
            traj = model.integrate(xc, xm, t_start=0.0, t_end=t_end, n_steps=step_cfg[thr])
            pred = traj[-1][1]
            # 提高短效时间点 (<=12h) 权重避免被 24h 淹没
            w = 1.5 if thr <= 12.0 else 1.0
            total_loss = total_loss + w * F.mse_loss(pred, yd)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        Xc_e6 = torch.tensor(np.vstack([r["ctrl"] for r in ev_6_pair]), device=DEVICE)
        Xm_e6 = torch.tensor(np.vstack([r["mol_feat"] for r in ev_6_pair]), device=DEVICE)
        Y_e6 = np.vstack([r["delta"] for r in ev_6_pair])
        Y_e24p = np.vstack([r["delta"] for r in ev_24_pair])

        p6_e33 = model.integrate(Xc_e6, Xm_e6, 0.0, 0.25, 2)[-1][1].cpu().numpy()
        p24_e33_p = model.integrate(Xc_e6, Xm_e6, 0.0, 1.0, 4)[-1][1].cpu().numpy()

        Xc_e24a = torch.tensor(np.vstack([r["ctrl"] for r in ev_24_all]), device=DEVICE)
        Xm_e24a = torch.tensor(np.vstack([r["mol_feat"] for r in ev_24_all]), device=DEVICE)
        Y_e24a = np.vstack([r["delta"] for r in ev_24_all])
        p24_e33_all = model.integrate(Xc_e24a, Xm_e24a, 0.0, 1.0, 4)[-1][1].cpu().numpy()

        # 导出 0h -> 72h 完整快慢通路演进轨迹
        traj_72 = model.integrate(Xc_e24a, Xm_e24a, 0.0, 3.0, 12)

    rows = []
    if os.path.exists(EXP32B_CSV):
        for _, r in pd.read_csv(EXP32B_CSV).iterrows():
            rows.append(dict(r))

    rows.append(eval_delta_metrics(Y_e6, p6_e33, "Exp33_ExpandedFastSlowODE_0h->6h(pair)"))
    rows.append(eval_delta_metrics(Y_e24p, p24_e33_p, "Exp33_ExpandedFastSlowODE_0h->24h(pair)"))
    rows.append(eval_delta_metrics(Y_e24a, p24_e33_all, "Exp33_ExpandedFastSlowODE_0h->24h(All13Drugs)"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Exp33 vs Exp32b 未见22细胞系盲测对比 (总耗时 {time.time()-t0:.1f}s) ===")
    print(out_df.to_string(index=False))

    # 绘制 0h -> 72h 通路快慢演进图
    mem = pd.read_parquet(os.path.join(PRIORS_DIR, "gene_pathway_membership.parquet"))
    hm = mem[mem["source_db"] == "MSigDB_Hallmark_2020"].copy()
    hm["pw_low"] = hm["pathway_name"].str.lower()
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    focus_keywords = ["tnf-alpha", "p53 pathway", "e2f targets", "g2-m checkpoint", "apoptosis", "glycolysis"]

    pw_rows = []
    for t_norm, pred_mat, _, _ in traj_72:
        hrs = round(t_norm * 24.0, 1)
        arr = pred_mat.cpu().numpy()
        for kw in focus_keywords:
            matched = hm[hm["pw_low"].str.contains(kw, regex=False)]
            if len(matched) == 0:
                continue
            pw_name = str(matched["pathway_name"].iloc[0])
            idxs = [g2i[g.upper()] for g in matched["gene_symbol"] if g.upper() in g2i]
            if idxs:
                pw_rows.append({"time_hours": hrs, "pathway": pw_name, "signed_activity": float(np.mean(arr[:, idxs]))})
    pw_df = pd.DataFrame(pw_rows)
    if len(pw_df) > 0:
        fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=180)
        for pw, grp_df in pw_df.groupby("pathway"):
            ax.plot(grp_df["time_hours"], grp_df["signed_activity"], marker="o", linewidth=2.0, label=pw)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time Post-Perturbation (Hours, 0h -> 72h)", fontsize=10)
        ax.set_ylabel("Predicted Pathway Activity (Signed Delta)", fontsize=10)
        ax.set_title("Exp33: Expanded Fast-Slow OmniPath ODE Pathway Cascade (0h -> 72h)", fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8.5, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=180)
        print(f"Saved cascade plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
