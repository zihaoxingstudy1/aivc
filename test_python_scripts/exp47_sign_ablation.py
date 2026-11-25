# -*- coding: utf-8 -*-
"""
exp47_sign_ablation.py

Does the AIVC cross-lineage gain come from the *sign* of the OmniPath prior
(activation vs inhibition), or merely from its topology and magnitude?

The prior is rebuilt from `data/priors/omnipath_raw_human.tsv` using exactly the
rules in pkb.fetch_omnipath_directed_edges + pgnn.build_gene_pathway_bipartite_tensors
(parquet reading is unavailable in this environment; the TSV is the upstream source
and the reconstruction is gated against the published edge counts 2,838 / 1,006).

Arms
----
  signed       : A_signed = A_pos - A_neg            reference; should reproduce v1.1
  unsigned_abs : |A_signed|                          sign stripped from the signed operator
  unsigned_sum : A_pos + A_neg                       both edge classes treated as activation
  shuffled     : signs permuted over non-zero cells, positive/negative counts preserved
  none         : causal footprint zeroed
  factorial    : +/- causal  x  +/- baseline phospho (exp28), reported separately

Run
---
  python test_python_scripts/exp47_sign_ablation.py --arms signed --seeds 1
  python test_python_scripts/exp47_sign_ablation.py --arms signed,unsigned_abs,unsigned_sum --seeds 5
"""

import os
import sys
import time
import types
import argparse

for _m in ("gseapy", "scanpy"):          # imported by project modules, blocked here
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "func_python_scripts"))
sys.path.insert(0, os.path.join(ROOT, "test_python_scripts"))

from exp27b_local_gated_5omics import LocalGated5OmicsPathwayGNN           # noqa: E402
from eval_calibrated_wmse_pds import (                                    # noqa: E402
    compute_pds_per_pert, compute_wmse_per_pert,
)

PRED_DIR = os.path.join(ROOT, "test_results", "sota_bench_preds")
TSV = os.path.join(ROOT, "data", "priors", "omnipath_raw_human.tsv")
EXP28_CKPT = os.path.join(ROOT, "pretrained", "aivc_5omics_pathway_gnn.pt")
OUT_CSV = os.path.join(ROOT, "test_results", "exp47_sign_ablation_results.csv")
OUT_TER = os.path.join(ROOT, "test_results", "exp47_sign_ablation_tertiles.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
SEED0 = 42
EXPECT_POS, EXPECT_NEG = 2838, 1006     # published counts inside the 3,603-gene universe


# ============================================================ prior reconstruction
def _norm(sym):
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    return "" if s in {"", "-", "NAN", "NONE", "NA"} else s


def _row_norm(mat):
    rs = mat.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return (mat / rs).astype(np.float32)


def build_prior_from_tsv(genes, tsv_path):
    g2i = {_norm(g): i for i, g in enumerate(genes)}
    n = len(genes)
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    src = df["source_genesymbol"].map(_norm).to_numpy()
    dst = df["target_genesymbol"].map(_norm).to_numpy()
    keep = (src != "") & (dst != "") & (src != dst)

    ce = pd.to_numeric(df.get("curation_effort", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    keep &= ce >= 1

    stim = pd.to_numeric(df.get("is_stimulation", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    inh = pd.to_numeric(df.get("is_inhibition", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    sign = np.where((stim == 1) & (inh == 0), 1, np.where((inh == 1) & (stim == 0), -1, 0))

    sub = pd.DataFrame({"src": src[keep], "dst": dst[keep], "sign": sign[keep]})
    sub = sub.drop_duplicates(subset=["src", "dst", "sign"])

    A_pos = np.zeros((n, n), dtype=np.float32)
    A_neg = np.zeros((n, n), dtype=np.float32)
    n_pos = n_neg = 0
    for s, d, g in zip(sub["src"].to_numpy(), sub["dst"].to_numpy(), sub["sign"].to_numpy()):
        i, j = g2i.get(s), g2i.get(d)
        if i is None or j is None or i == j:
            continue
        if g == 1:
            A_pos[j, i] = 1.0
            n_pos += 1
        elif g == -1:
            A_neg[j, i] = 1.0
            n_neg += 1
    return _row_norm(A_pos), _row_norm(A_neg), n_pos, n_neg


# ============================================================ model (exp31 copy)
class LineageInvariantPathwayNet(nn.Module):
    def __init__(self, n_genes, n_teacher, k_prog, U_prog, stat_dim=5, hid=128):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.ridge_scale = nn.Parameter(torch.ones(1))
        self.target_gate = nn.Sequential(nn.Linear(stat_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)
        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.causal_enc = nn.Sequential(nn.Linear(k_prog, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.25))
        self.prog_head = nn.Sequential(
            nn.Linear(64 + hid + 32 + stat_dim, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.prog_head[-1].weight)
        nn.init.zeros_(self.prog_head[-1].bias)

    def forward(self, x_ctrl, x_teach, x_ridge, x_tstat, x_causal):
        gate = 1.0 + 0.5 * torch.tanh(self.target_gate(x_tstat))
        h = torch.cat([self.ctrl_enc(x_ctrl), self.teach_enc(x_teach),
                       self.causal_enc(x_causal), x_tstat], dim=-1)
        return gate * (self.ridge_scale * x_ridge) + 0.5 * (self.prog_head(h) @ self.U_prog)


@torch.no_grad()
def predict_lineage_prot_phos(lineage_ctrl_dict, target_genes):
    ck = torch.load(EXP28_CKPT, map_location="cpu", weights_only=False)
    top_rna = [str(g).upper() for g in ck["top_rna"]]
    top_prot = [str(g).upper() for g in ck["top_prot"]]
    top_phos = [str(g).upper() for g in ck["top_phos"]]

    net = LocalGated5OmicsPathwayGNN(
        n_rna=len(top_rna), n_mut=len(ck["driver_genes"]), n_mir=len(ck["top_mir"]),
        rppa_cis=ck["rppa_cis"], prot_cis=ck["prot_cis"], phos_cis=ck["phos_cis"],
        A_tf=ck["A_tf"], A_gp=ck["A_gp"], M_rppa=ck["M_rppa"], M_prot=ck["M_prot"],
        M_phos=ck["M_phos"], hid=160,
    ).to(DEVICE)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    sc_g2i = {str(g).upper(): i for i, g in enumerate(target_genes)}
    prot_g2i = {g: i for i, g in enumerate(top_prot)}
    phos_g2i = {g: i for i, g in enumerate(top_phos)}

    out = {}
    for lname, mu_vec in lineage_ctrl_dict.items():
        x_in = np.zeros((1, len(top_rna)), dtype=np.float32)
        mu_z = (mu_vec - mu_vec.mean()) / (mu_vec.std() + 1e-6)
        for j, g in enumerate(top_rna):
            idx = sc_g2i.get(g)
            if idx is not None:
                x_in[0, j] = float(mu_z[idx])
        xt = torch.tensor(x_in, device=DEVICE)
        pred_prot = net.forward_prot(xt, None, None).squeeze(0).cpu().numpy()
        pred_phos = net.forward_phos(xt, None, None).squeeze(0).cpu().numpy()
        g_prot = np.zeros(len(target_genes), dtype=np.float32)
        g_phos = np.zeros(len(target_genes), dtype=np.float32)
        for idx, g in enumerate(target_genes):
            gu = str(g).upper()
            if gu in prot_g2i:
                g_prot[idx] = float(pred_prot[prot_g2i[gu]])
            if gu in phos_g2i:
                g_phos[idx] = float(pred_phos[phos_g2i[gu]])
        out[lname] = (g_prot, g_phos)
    return out


# ============================================================ metrics
def _safe_r(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def per_perturbation_metrics(Y, P, k=20):
    n = Y.shape[0]
    mean_r = np.empty(n); top_r = np.empty(n); dir_acc = np.empty(n); mse = np.empty(n)
    for i in range(n):
        dt, dp = Y[i], P[i]
        mean_r[i] = _safe_r(dt, dp)
        mse[i] = float(np.mean((dt - dp) ** 2))
        idx = np.argsort(np.abs(dt))[-k:]
        top_r[i] = _safe_r(dt[idx], dp[idx])
        dir_acc[i] = float(np.mean(np.sign(dt[idx]) == np.sign(dp[idx])))
    return mean_r, top_r, dir_acc, mse


def pds(P, T):
    """Official implementation (row-wise cosine distance, mid-rank ties)."""
    return float(np.mean(compute_pds_per_pert(T, P)))


def wmse(P, T):
    """Official implementation (|delta|-weighted MSE)."""
    return float(np.mean(compute_wmse_per_pert(T, P)))


# ============================================================ features / training
def build_causal_matrix(A, U_prog):
    return ((A + 0.5 * (A @ A)) @ U_prog.T).astype(np.float32)


def build_features(plist, lnames, ridge_preds, esm_mat, coexp_mat, ctrls,
                   g2i, pp, causal_mat, use_causal=True, use_phos=True):
    ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
    for idx, (ln, p) in enumerate(zip(lnames, plist)):
        mc = ctrls[ln]
        ctrl_l.append(mc)
        teach_l.append(np.concatenate([coexp_mat[idx], esm_mat[idx]]))
        gi = g2i.get(p, -1)
        if gi >= 0:
            g_prot, g_phos = pp[ln]
            raw_e = float(mc[gi])
            z_e = float((mc[gi] - mc.mean()) / (mc.std() + 1e-6))
            pr_v, ph_v = (float(g_prot[gi]), float(g_phos[gi])) if use_phos else (0.0, 0.0)
            c_vec = causal_mat[gi] if use_causal else np.zeros(64, dtype=np.float32)
        else:
            raw_e = z_e = pr_v = ph_v = 0.0
            c_vec = np.zeros(64, dtype=np.float32)
        tstat_l.append([raw_e, z_e, pr_v, ph_v, float(np.log1p(np.linalg.norm(ridge_preds[idx])))])
        causal_l.append(np.asarray(c_vec, dtype=np.float32))
    return (np.asarray(ctrl_l, dtype=np.float32), np.asarray(teach_l, dtype=np.float32),
            ridge_preds.astype(np.float32), np.asarray(tstat_l, dtype=np.float32),
            np.asarray(causal_l, dtype=np.float32))


def train_predict(f_tr, f_ev, Y_tr, n_genes, U_prog, seeds):
    ctrl_tr, teach_tr, r_tr, tstat_tr, caus_tr = f_tr
    ctrl_ev, teach_ev, r_ev, tstat_ev, caus_ev = f_ev
    preds = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        model = LineageInvariantPathwayNet(n_genes, teach_tr.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
        dl = DataLoader(TensorDataset(*[torch.tensor(x) for x in
                                        (ctrl_tr, teach_tr, r_tr, tstat_tr, caus_tr)], torch.tensor(Y_tr)),
                        batch_size=512, shuffle=True)
        model.train()
        for _ in range(30):
            for bc, bt, br, bts, bca, by in dl:
                bc, bt, br, bts, bca, by = [x.to(DEVICE) for x in (bc, bt, br, bts, bca, by)]
                opt.zero_grad()
                F.mse_loss(model(bc, bt, br, bts, bca), by).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            pev = model(*[torch.tensor(x, device=DEVICE) for x in
                          (ctrl_ev, teach_ev, r_ev, tstat_ev, caus_ev)])
        preds.append(pev.cpu().numpy().astype(np.float32))
    return preds


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="signed")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    b = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    genes = [str(g) for g in b["target_genes"]]
    g2i = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    ev_keys = [str(k) for k in b["ev_keys"]]
    tr_perts = [str(p) for p in b["tr_perts"]]
    tr_lin = np.array([str(x) for x in b["tr_lineages"]])
    Y_tr, Y_ev = b["Y_tr"], b["Y_ev"]
    ctrls = {str(k): v for k, v in zip(b["ctrl_names"], b["ctrl_vecs"])}

    pred_ridge_ev = np.load(os.path.join(PRED_DIR, "pred_Linear_Ridge.npy"))
    vocab = sorted(set(tr_perts)); vidx = {t: i for i, t in enumerate(vocab)}
    Xl = np.zeros((len(tr_perts), len(vocab)), dtype=np.float32)
    for ri, p in enumerate(tr_perts):
        Xl[ri, vidx[p]] = 1.0
    pred_ridge_tr = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lin):
        m_out = tr_lin == ln; m_in = ~m_out
        if m_in.sum():
            pred_ridge_tr[m_out] = Ridge(alpha=50.0).fit(Xl[m_in], Y_tr[m_in]).predict(Xl[m_out]).astype(np.float32)

    U_prog = TruncatedSVD(n_components=64, random_state=SEED0).fit(Y_tr).components_.astype(np.float32)

    print("[exp47] reconstructing OmniPath prior from raw TSV ...", flush=True)
    A_pos, A_neg, n_pos, n_neg = build_prior_from_tsv(genes, TSV)
    ok = (n_pos == EXPECT_POS) and (n_neg == EXPECT_NEG)
    print(f"[exp47] edges mapped: activation={n_pos} (expect {EXPECT_POS}), "
          f"inhibition={n_neg} (expect {EXPECT_NEG}) -> {'MATCH' if ok else 'MISMATCH'}", flush=True)
    A_signed = (A_pos - A_neg).astype(np.float32)

    print("[exp47] inferring baseline protein/phosphosite states ...", flush=True)
    pp = predict_lineage_prot_phos(ctrls, genes)

    rng = np.random.RandomState(7)
    nz = A_signed != 0
    vals = A_signed[nz].copy()
    A_sh = np.zeros_like(A_signed); A_sh[nz] = vals[rng.permutation(len(vals))]

    pool = {
        "signed": dict(A=A_signed, use_causal=True, use_phos=True),
        "unsigned_abs": dict(A=np.abs(A_signed), use_causal=True, use_phos=True),
        "unsigned_sum": dict(A=(A_pos + A_neg).astype(np.float32), use_causal=True, use_phos=True),
        "shuffled": dict(A=A_sh, use_causal=True, use_phos=True),
        "none": dict(A=None, use_causal=False, use_phos=True),
        "causal_only": dict(A=A_signed, use_causal=True, use_phos=False),
        "phospho_only": dict(A=None, use_causal=False, use_phos=True),
        "neither": dict(A=None, use_causal=False, use_phos=False),
    }
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in pool]
    missing = [a for a in args.arms.split(",") if a.strip() and a.strip() not in pool]
    if missing:
        raise SystemExit(f"unknown arms: {missing}")

    seeds = [42, 101, 202, 303, 404][: args.seeds]
    order = np.argsort(np.linalg.norm(Y_ev, axis=1))
    q = np.array_split(order, 3)
    tert = {"weak": q[0], "moderate": q[1], "strong": q[2]}

    rows, trows = [], []
    for name in arms:
        cfg = pool[name]
        t1 = time.time()
        cmat = np.zeros((n_genes, 64), dtype=np.float32) if cfg["A"] is None \
            else build_causal_matrix(cfg["A"], U_prog)
        f_tr = build_features(tr_perts, tr_lin.tolist(), pred_ridge_tr,
                              b["X_esm_tr"], b["X_coexp_tr"], ctrls, g2i, pp,
                              cmat, cfg["use_causal"], cfg["use_phos"])
        f_ev = build_features(ev_keys, ["RPE1"] * len(ev_keys), pred_ridge_ev,
                              b["X_esm_ev"], b["X_coexp_ev"], ctrls, g2i, pp,
                              cmat, cfg["use_causal"], cfg["use_phos"])
        preds = train_predict(f_tr, f_ev, Y_tr, n_genes, U_prog, seeds)
        for sd, P in zip(seeds, preds):
            np.save(os.path.join(ROOT, "test_results", f"exp47_pred_{name}_s{sd}.npy"), P)

        for sd, P in zip(seeds, preds):
            mr, trr, da, ms = per_perturbation_metrics(Y_ev, P, 20)
            rows.append({"arm": name, "seed": sd, "mean_r": mr.mean(), "top20_r": trr.mean(),
                         "dir_acc": da.mean(), "mse": ms.mean(), "wmse": wmse(P, Y_ev),
                         "pds": pds(P, Y_ev), "sec": round(time.time() - t1, 1)})
            for tname, idx in tert.items():
                trows.append({"arm": name, "seed": sd, "tertile": tname,
                              "top20_r": float(trr[idx].mean()), "mean_r": float(mr[idx].mean())})

        agg = pd.DataFrame([r for r in rows if r["arm"] == name])
        print(f"[{name:13s}] meanR={agg.mean_r.mean():.4f} Top20R={agg.top20_r.mean():.4f} "
              f"DirAcc={agg.dir_acc.mean():.4f} MSE={agg.mse.mean():.6f} WMSE={agg.wmse.mean():.5f} "
              f"PDS={agg.pds.mean():.4f} ({time.time()-t1:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        res = pd.concat([pd.read_csv(OUT_CSV), res]).drop_duplicates(subset=["arm", "seed"], keep="last")
    res.to_csv(OUT_CSV, index=False)
    ter = pd.DataFrame(trows)
    if os.path.exists(OUT_TER):
        ter = pd.concat([pd.read_csv(OUT_TER), ter]).drop_duplicates(subset=["arm", "seed", "tertile"], keep="last")
    ter.to_csv(OUT_TER, index=False)
    print(f"\n[exp47] wrote {OUT_CSV} and {OUT_TER} (total {time.time()-t0:.0f}s)")
    print(res.groupby("arm")[["mean_r", "top20_r", "dir_acc", "mse", "wmse", "pds"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()