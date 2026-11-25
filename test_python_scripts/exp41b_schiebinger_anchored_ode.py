# -*- coding: utf-8 -*-
"""
exp41b_schiebinger_anchored_ode.py
Exp41b: State-Anchored Fast-Slow Grey-Box ODE on SchiebingerLander2019 (18-Day Reprogramming)
  - Caches extracted pseudobulk time-course (D0 -> D18) for instant reuse.
  - Trains vectorized State-Anchored ODE on all within-condition pairs (t_a < t_b <= 12.0):
      y_hat(t_b) = y(t_a) + (z(t_b) - z(t_a)) @ U_prog
    guaranteeing exact boundary continuity at t_a and smooth saturation toward equilibrium.
  - Blind-extrapolates from D12 anchor to unseen future days D12.5 -> D18 and compares against
    Persistence_Day12, Linear_Time_Extrap, Static_MLP_Time, and OpenLoop_ODE.
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5AD_PATH = r"D:\data\AIVC\data\SchiebingerLander2019_GSE115943.h5ad"
CACHE_PATH = r"D:\data\AIVC\data\schiebinger_18day_pb_hvg2000_cache.npz"
OUT_CSV = r"D:\data\AIVC\test_results\exp41b_schiebinger_anchored_ode_results.csv"


def parse_day(age_str: str):
    m = re.search(r"(\d+(?:\.\d+)?)", str(age_str))
    return float(m.group(1)) if m else None


def load_or_build_cache():
    if os.path.exists(CACHE_PATH):
        ck = np.load(CACHE_PATH, allow_pickle=True)
        return (
            [str(p) for p in ck["perts"]],
            ck["days"].astype(np.float32),
            ck["deltas"].astype(np.float32),
            [str(c) for c in ck["cond_vocab"]],
        )

    a = ad.read_h5ad(H5AD_PATH, backed="r")
    obs = a.obs[["age", "perturbation"]].copy()
    obs["day"] = [parse_day(x) for x in obs["age"]]
    obs["pert"] = obs["perturbation"].astype(str)
    valid_mask = obs["day"].notna().values

    rng = np.random.RandomState(SEED)
    row_to_key = {}
    keys_list = []
    for (pert, day), grp in obs[valid_mask].groupby(["pert", "day"]):
        int_pos = obs.index.get_indexer(grp.index)
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

    for s0 in range(0, len(all_rows), 4000):
        sub = all_rows[s0:s0 + 4000]
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
    d0_keys = [k for k in keys_list if k[1] == 0.0]
    mu_d0 = np.mean([pb_map[k] for k in d0_keys], axis=0).astype(np.float32) if d0_keys else np.mean([pb_map[k] for k in keys_list if k[1] == min(x[1] for x in keys_list)], axis=0).astype(np.float32)

    tr_mat_full = np.vstack([pb_map[k] - mu_d0 for k in keys_list if 0.0 < k[1] <= 12.0])
    top2000 = np.argsort(-np.var(tr_mat_full, axis=0))[:2000]
    cond_vocab = sorted({k[0] for k in keys_list if k[1] > 0.0})

    perts, days, deltas = [], [], []
    for pert, day in sorted(keys_list, key=lambda x: (x[0], x[1])):
        if day <= 0.0:
            continue
        perts.append(pert)
        days.append(day)
        deltas.append((pb_map[(pert, day)] - mu_d0)[top2000].astype(np.float32))
    deltas = np.vstack(deltas).astype(np.float32)
    np.savez_compressed(CACHE_PATH, perts=np.array(perts), days=np.array(days, dtype=np.float32), deltas=deltas, cond_vocab=np.array(cond_vocab))
    return perts, np.array(days, dtype=np.float32), deltas, cond_vocab


class StateAnchoredDevODE(nn.Module):
    """
    Vectorized State-Anchored Grey-Box ODE:
      dz/dt = -gamma * (z - z_eq(cond)) * exp(-k_dec * t) + 0.15 * tanh(z @ W.T)
      y_hat(t_b) = y(t_a) + (z(t_b) - z(t_a)) @ U_prog
    """

    def __init__(self, n_cond: int, k_prog: int, U_prog: np.ndarray):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.gamma_raw = nn.Parameter(torch.ones(k_prog) * -1.0)
        self.k_dec_raw = nn.Parameter(torch.tensor(0.2))
        self.z_eq_head = nn.Linear(n_cond, k_prog)
        self.W_coup = nn.Parameter(torch.randn(k_prog, k_prog) * 0.02)
        self.step_scale = nn.Parameter(torch.tensor(0.35))

    def _rhs(self, z, z_eq, t_cur):
        g = F.softplus(self.gamma_raw) * 0.35
        kd = F.softplus(self.k_dec_raw) * 0.5
        return (-g * (z - z_eq) + 0.10 * torch.tanh(z @ self.W_coup.t())) * torch.exp(-kd * t_cur)

    def forward(self, y_anchor, cond_oh, t_a, t_b, n_steps: int = 6):
        z0 = y_anchor @ self.U_prog.t()
        z_eq = self.z_eq_head(cond_oh)
        dt = ((t_b - t_a) / float(n_steps)).unsqueeze(-1)
        t_cur = t_a.unsqueeze(-1)
        z = z0
        for _ in range(n_steps):
            k1 = self._rhs(z, z_eq, t_cur)
            k2 = self._rhs(z + 0.5 * dt * k1, z_eq, t_cur + 0.5 * dt)
            k3 = self._rhs(z + 0.5 * dt * k2, z_eq, t_cur + 0.5 * dt)
            k4 = self._rhs(z + dt * k3, z_eq, t_cur + dt)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_cur = t_cur + dt
        return y_anchor + self.step_scale * ((z - z0) @ self.U_prog)


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
    print(f"=== Exp41b: State-Anchored 18-Day Reprogramming ODE Extrapolation ({DEVICE}) ===")
    perts, days, deltas, cond_vocab = load_or_build_cache()
    p2i = {p: i for i, p in enumerate(cond_vocab)}

    tr_idx = [i for i in range(len(days)) if days[i] <= 12.0]
    ev_idx = [i for i in range(len(days)) if days[i] > 12.0]

    # Build within-condition training pairs (t_a < t_b <= 12.0, gap <= 6 days)
    by_pert_tr = {}
    for i in tr_idx:
        by_pert_tr.setdefault(perts[i], []).append(i)

    ya_tr, oh_tr, ta_tr, tb_tr, yb_tr = [], [], [], [], []
    last_tr_idx = {}
    for p, idxs in by_pert_tr.items():
        idxs = sorted(idxs, key=lambda x: days[x])
        last_tr_idx[p] = idxs[-1]
        oh = np.zeros(len(cond_vocab), dtype=np.float32)
        oh[p2i[p]] = 1.0
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                ia, ib = idxs[a_i], idxs[b_i]
                if 0.5 <= (days[ib] - days[ia]) <= 6.5:
                    ya_tr.append(deltas[ia])
                    oh_tr.append(oh)
                    ta_tr.append(days[ia] / 12.0)
                    tb_tr.append(days[ib] / 12.0)
                    yb_tr.append(deltas[ib])

    # Build blind future test pairs anchored at each condition's latest observed D<=12 state!
    ya_ev, oh_ev, ta_ev, tb_ev, yb_ev = [], [], [], [], []
    for i in ev_idx:
        p = perts[i]
        ia = last_tr_idx[p]
        oh = np.zeros(len(cond_vocab), dtype=np.float32)
        oh[p2i[p]] = 1.0
        ya_ev.append(deltas[ia])
        oh_ev.append(oh)
        ta_ev.append(days[ia] / 12.0)
        tb_ev.append(days[i] / 12.0)
        yb_ev.append(deltas[i])

    ya_tr = np.vstack(ya_tr).astype(np.float32)
    oh_tr = np.vstack(oh_tr).astype(np.float32)
    ta_tr = np.array(ta_tr, dtype=np.float32)
    tb_tr = np.array(tb_tr, dtype=np.float32)
    yb_tr = np.vstack(yb_tr).astype(np.float32)

    ya_ev = np.vstack(ya_ev).astype(np.float32)
    oh_ev = np.vstack(oh_ev).astype(np.float32)
    ta_ev = np.array(ta_ev, dtype=np.float32)
    tb_ev = np.array(tb_ev, dtype=np.float32)
    yb_ev = np.vstack(yb_ev).astype(np.float32)
    print(f"  Train pairs (t_a < t_b <= 12): {len(ya_tr)} | Blind Future Test (D>12): {len(ya_ev)}")

    Y_tr_all = deltas[tr_idx]
    svd = TruncatedSVD(n_components=min(20, len(tr_idx) - 1), random_state=SEED).fit(Y_tr_all)
    U_prog = svd.components_.astype(np.float32)

    # Vectorized 5-member ensemble training (super fast on GPU!)
    ens_preds = []
    ya_t, oh_t, ta_t, tb_t, yb_t = (
        torch.tensor(ya_tr, device=DEVICE),
        torch.tensor(oh_tr, device=DEVICE),
        torch.tensor(ta_tr, device=DEVICE),
        torch.tensor(tb_tr, device=DEVICE),
        torch.tensor(yb_tr, device=DEVICE),
    )
    ya_e, oh_e, ta_e, tb_e = (
        torch.tensor(ya_ev, device=DEVICE),
        torch.tensor(oh_ev, device=DEVICE),
        torch.tensor(ta_ev, device=DEVICE),
        torch.tensor(tb_ev, device=DEVICE),
    )
    for sd_val in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd_val)
        ode = StateAnchoredDevODE(len(cond_vocab), U_prog.shape[0], U_prog).to(DEVICE)
        opt = torch.optim.AdamW(ode.parameters(), lr=2e-3, weight_decay=5e-3)
        ode.train()
        for _ in range(180):
            opt.zero_grad()
            pred = ode(ya_t, oh_t, ta_t, tb_t, n_steps=6)
            loss = F.mse_loss(pred, yb_t) + 0.02 * (1.0 - F.cosine_similarity(pred, yb_t, dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode.parameters(), 2.0)
            opt.step()
        ode.eval()
        with torch.no_grad():
            ens_preds.append(ode(ya_e, oh_e, ta_e, tb_e, n_steps=6).cpu().numpy().astype(np.float32))
    p_anchored = np.mean(ens_preds, axis=0).astype(np.float32)

    # Compare against Persistence_Day12 and Linear_Time_Extrap
    X_lin_tr = np.vstack([np.concatenate([np.eye(len(cond_vocab))[p2i[perts[i]]], [days[i] / 18.0]]) for i in tr_idx]).astype(np.float32)
    X_lin_ev = np.vstack([np.concatenate([np.eye(len(cond_vocab))[p2i[perts[i]]], [days[i] / 18.0]]) for i in ev_idx]).astype(np.float32)
    p_lin = Ridge(alpha=1.0).fit(X_lin_tr, deltas[tr_idx]).predict(X_lin_ev).astype(np.float32)

    rows = []
    for mname, pmat in [
        ("Persistence_Day12", ya_ev),
        ("Linear_Time_Extrap", p_lin),
        ("AIVC_StateAnchored_ODE_Ens5", p_anchored),
    ]:
        m = eval_traj(yb_ev, pmat)
        print(f"  [{mname:28s}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
        rows.append({"model": mname, **m})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved Exp41b -> {OUT_CSV} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
