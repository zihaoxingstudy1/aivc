# -*- coding: utf-8 -*-
"""
exp41_schiebinger_18day_ode.py
Exp41: Cross-Species, Long-Horizon (18-Day) Developmental Reprogramming ODE Extrapolation
  - Streams SchiebingerLander2019_GSE115943.h5ad (259,155 cells, D0 -> D18 iPSC reprogramming).
  - Trains Fast-Slow Grey-Box ODE strictly on early-to-mid days (day <= 12.0).
  - Blind-extrapolates forward to unseen late days (day > 12.0, up to D18) and compares against
    Last-Observed Persistence, Linear Time Extrapolation, and Static-MLP(t).
"""

import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5AD_PATH = r"D:\data\AIVC\data\SchiebingerLander2019_GSE115943.h5ad"
OUT_CSV = r"D:\data\AIVC\test_results\exp41_schiebinger_18day_ode_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp41_schiebinger_18day_ode_plot.png"


def parse_day(age_str: str):
    m = re.search(r"(\d+(?:\.\d+)?)", str(age_str))
    return float(m.group(1)) if m else None


class FastSlowDevODE(nn.Module):
    """Fast-Slow Grey-Box Latent ODE for multi-week developmental reprogramming trajectories."""

    def __init__(self, n_cond: int, k_prog: int, U_prog: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.gamma_fast = nn.Parameter(torch.ones(k_prog) * 0.8)
        self.gamma_slow = nn.Parameter(torch.ones(k_prog) * -1.0)
        self.k_dec = nn.Parameter(torch.tensor(-0.5))
        self.u_fast = nn.Linear(n_cond, k_prog)
        self.u_slow = nn.Linear(n_cond, k_prog)
        self.W_f2s = nn.Parameter(torch.randn(k_prog, k_prog) * 0.04)
        self.W_s2s = nn.Parameter(torch.randn(k_prog, k_prog) * 0.03)

    def _rhs(self, zf, zs, uf, us, t_val):
        gf = F.softplus(self.gamma_fast)
        gs = F.softplus(self.gamma_slow) * 0.25
        kd = F.softplus(self.k_dec) * 0.3
        dzf = -gf * zf + uf * torch.exp(-kd * t_val)
        dzs = -gs * zs + 0.4 * torch.tanh(zf @ self.W_f2s.t() + zs @ self.W_s2s.t()) + 0.5 * us
        return dzf, dzs

    def integrate(self, cond_oh: torch.Tensor, t_end_norm: float, n_steps: int = 8) -> torch.Tensor:
        uf = self.u_fast(cond_oh)
        us = self.u_slow(cond_oh)
        B = cond_oh.shape[0]
        k = self.U_prog.shape[0]
        zf = torch.zeros((B, k), device=cond_oh.device)
        zs = torch.zeros((B, k), device=cond_oh.device)
        dt = float(t_end_norm) / float(max(n_steps, 1))
        t_cur = 0.0
        for _ in range(max(n_steps, 1)):
            k1f, k1s = self._rhs(zf, zs, uf, us, t_cur)
            k2f, k2s = self._rhs(zf + 0.5 * dt * k1f, zs + 0.5 * dt * k1s, uf, us, t_cur + 0.5 * dt)
            k3f, k3s = self._rhs(zf + 0.5 * dt * k2f, zs + 0.5 * dt * k2s, uf, us, t_cur + 0.5 * dt)
            k4f, k4s = self._rhs(zf + dt * k3f, zs + dt * k3s, uf, us, t_cur + dt)
            zf = zf + (dt / 6.0) * (k1f + 2 * k2f + 2 * k3f + k4f)
            zs = zs + (dt / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
            t_cur += dt
        return (zf + zs) @ self.U_prog


def eval_traj(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def main():
    t0 = time.time()
    print(f"=== Exp41: SchiebingerLander2019 18-Day Reprogramming ODE Extrapolation ({DEVICE}) ===")
    a = ad.read_h5ad(H5AD_PATH, backed="r")
    obs = a.obs[["age", "perturbation"]].copy()
    obs["day"] = [parse_day(x) for x in obs["age"]]
    obs["pert"] = obs["perturbation"].astype(str)
    valid_mask = obs["day"].notna().values

    # Group by (pert, day), sample up to 300 cells per group and read sequentially
    rng = np.random.RandomState(SEED)
    row_to_key = {}
    keys_list = []
    for (pert, day), grp in obs[valid_mask].groupby(["pert", "day"]):
        pos = grp.index
        # Integer positions in obs
        int_pos = obs.index.get_indexer(pos)
        if len(int_pos) < 30:
            continue
        if len(int_pos) > 300:
            int_pos = rng.choice(int_pos, size=300, replace=False)
        key = (str(pert), float(day))
        keys_list.append(key)
        for ri in int_pos:
            row_to_key[int(ri)] = key

    all_rows = sorted(row_to_key.keys())
    n_all_genes = a.shape[1]
    sums = {k: np.zeros(n_all_genes, dtype=np.float64) for k in keys_list}
    cnts = {k: 0 for k in keys_list}

    chunk_size = 4000
    for s0 in range(0, len(all_rows), chunk_size):
        sub = all_rows[s0:s0 + chunk_size]
        X_sub = a.X[sub, :]
        if hasattr(X_sub, "toarray"):
            X_sub = X_sub.toarray()
        X_sub = np.log1p(np.clip(np.asarray(X_sub, dtype=np.float32), 0.0, None))
        for li, ri in enumerate(sub):
            k = row_to_key[ri]
            sums[k] += X_sub[li]
            cnts[k] += 1
    a.file.close()

    pb_map = {k: (sums[k] / cnts[k]).astype(np.float32) for k in keys_list}
    # Find D0 baseline
    d0_keys = [k for k in keys_list if k[1] == 0.0]
    if d0_keys:
        mu_d0 = np.mean([pb_map[k] for k in d0_keys], axis=0).astype(np.float32)
    else:
        min_d = min(k[1] for k in keys_list)
        mu_d0 = np.mean([pb_map[k] for k in keys_list if k[1] == min_d], axis=0).astype(np.float32)

    # Select top 2000 most variable genes across early/mid timepoints (day <= 12.0, zero test leakage)
    tr_mat_full = np.vstack([pb_map[k] - mu_d0 for k in keys_list if k[1] > 0.0 and k[1] <= 12.0])
    top2000 = np.argsort(-np.var(tr_mat_full, axis=0))[:2000]

    perts_sorted = sorted({k[0] for k in keys_list if k[1] > 0.0})
    p2i = {p: i for i, p in enumerate(perts_sorted)}

    tr_items, ev_items = [], []
    last_tr_by_pert = {}
    for pert, day in sorted(keys_list, key=lambda x: (x[0], x[1])):
        if day <= 0.0:
            continue
        dv = (pb_map[(pert, day)] - mu_d0)[top2000].astype(np.float32)
        oh = np.zeros(len(perts_sorted), dtype=np.float32)
        oh[p2i[pert]] = 1.0
        item = {"pert": pert, "day": day, "oh": oh, "delta": dv}
        if day <= 12.0:
            tr_items.append(item)
            last_tr_by_pert[pert] = dv
        else:
            ev_items.append(item)

    print(f"  [Schiebinger 18-Day] Conditions={perts_sorted} | Train points (D<=12)={len(tr_items)} | Blind Future Test (D>12)={len(ev_items)}")
    Y_tr = np.vstack([x["delta"] for x in tr_items]).astype(np.float32)
    Y_ev = np.vstack([x["delta"] for x in ev_items]).astype(np.float32)
    max_day = max(x["day"] for x in ev_items)

    # 1. Last-Observed Persistence (D12 -> D12.5..D18)
    p_persist = np.vstack([last_tr_by_pert.get(x["pert"], Y_tr.mean(axis=0)) for x in ev_items]).astype(np.float32)

    # 2. Linear Time Extrapolation (Ridge on [oh, day/18])
    X_lin_tr = np.vstack([np.concatenate([x["oh"], [x["day"] / max_day]]) for x in tr_items]).astype(np.float32)
    X_lin_ev = np.vstack([np.concatenate([x["oh"], [x["day"] / max_day]]) for x in ev_items]).astype(np.float32)
    p_lin = Ridge(alpha=1.0).fit(X_lin_tr, Y_tr).predict(X_lin_ev).astype(np.float32)

    # 3. Static-MLP(t)
    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(X_lin_tr.shape[1], 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2000)).to(DEVICE)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(200):
        opt_m.zero_grad()
        F.mse_loss(mlp(torch.tensor(X_lin_tr, device=DEVICE)), torch.tensor(Y_tr, device=DEVICE)).backward()
        opt_m.step()
    mlp.eval()
    with torch.no_grad():
        p_mlp = mlp(torch.tensor(X_lin_ev, device=DEVICE)).cpu().numpy().astype(np.float32)

    # 4. AIVC Fast-Slow Grey-Box Developmental ODE (5-Member Ensemble)
    svd = TruncatedSVD(n_components=min(24, len(tr_items) - 1), random_state=SEED).fit(Y_tr)
    U_prog = svd.components_.astype(np.float32)

    ens_ode = []
    for sd_val in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd_val)
        ode = FastSlowDevODE(len(perts_sorted), U_prog.shape[0], U_prog).to(DEVICE)
        opt = torch.optim.AdamW(ode.parameters(), lr=3e-3, weight_decay=2e-3)
        ode.train()
        for ep in range(220):
            opt.zero_grad()
            loss = 0.0
            for item in tr_items:
                oh_t = torch.tensor(item["oh"], device=DEVICE).unsqueeze(0)
                y_t = torch.tensor(item["delta"], device=DEVICE).unsqueeze(0)
                n_st = max(2, int(round(item["day"])))
                pred = ode.integrate(oh_t, item["day"] / 12.0, n_steps=n_st)
                loss = loss + F.mse_loss(pred, y_t)
            (loss / len(tr_items)).backward()
            torch.nn.utils.clip_grad_norm_(ode.parameters(), 2.0)
            opt.step()
        ode.eval()
        with torch.no_grad():
            preds_ev = []
            for item in ev_items:
                oh_t = torch.tensor(item["oh"], device=DEVICE).unsqueeze(0)
                n_st = max(4, int(round(item["day"])))
                preds_ev.append(ode.integrate(oh_t, item["day"] / 12.0, n_steps=n_st).squeeze(0).cpu().numpy())
            ens_ode.append(np.vstack(preds_ev).astype(np.float32))
    p_ode = np.mean(ens_ode, axis=0).astype(np.float32)

    rows = []
    for mname, pmat in [
        ("Persistence_Day12", p_persist),
        ("Linear_Time_Extrap", p_lin),
        ("Static_MLP_Time", p_mlp),
        ("AIVC_FastSlow_ODE_Ens5", p_ode),
    ]:
        m = eval_traj(Y_ev, pmat)
        print(f"  [{mname:24s}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
        rows.append({"model": mname, **m})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
    x = np.arange(len(res_df))
    w = 0.26
    ax.bar(x - w, res_df["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, res_df["top20_pearson"], width=w, label="Top20 DEG Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, res_df["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in res_df["model"]], fontsize=8.5)
    ax.set_title("Exp41: Cross-Species 18-Day Reprogramming Future Extrapolation\n(Train D0-D12 -> Blind Future D12.5-D18, SchiebingerLander2019)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp41 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
