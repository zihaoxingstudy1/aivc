# -*- coding: utf-8 -*-
"""
exp33b_mixseq_fast_slow_clean.py
Exp33b: 纯净同平台 MIXseq 快慢双尺度世界模型 (无跨平台 517 基因缺失噪声)
整合 MIXseq 主体 (13药 6h+24h) + 隐藏 hash_tag 五时间点 (3h, 6h, 12h, 24h, 48h)，
在完全相同的 22 个未见细胞系盲测集上直接对比 Exp32b 与静态基线。
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp33_expanded_fast_slow_temporal_world_model import (
    FastSlowOmniPathODE, build_unified_mol_bank, eval_delta_metrics
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
EXP32B_CSV = r"D:\data\AIVC\test_results\exp32b_temporal_world_model_results.csv"
OUT_CSV = r"D:\data\AIVC\test_results\exp33b_mixseq_fast_slow_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp33b_mixseq_fast_slow_cascade.png"


def main():
    t0 = time.time()
    print(f"=== Exp33b: 纯净同平台 MIXseq 快慢双尺度世界模型 ({DEVICE}) ===")

    ad_mix = sc.read_h5ad(os.path.join(DATA_DIR, "MIXseq_hvg2000.h5ad"))
    genes = list(ad_mix.var_names)
    n_genes = len(genes)
    obs_m = ad_mix.obs[["cell_line", "time", "perturbation", "dose_value", "hash_tag"]].copy()
    for c in obs_m.columns:
        obs_m[c] = obs_m[c].astype(str)
    X_m = ad_mix.X.toarray().astype(np.float32) if hasattr(ad_mix.X, "toarray") else np.asarray(ad_mix.X, dtype=np.float32)
    del ad_mix

    samples = []
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
        c_base = ctrl_6.get(cl) if tm == "6" else ctrl_24.get(cl)
        if c_base is None:
            continue
        dv = (X_m[idx].mean(axis=0) - c_base).astype(np.float32)
        samples.append({"src": "MIXseq", "cl": cl, "drug": drug, "dose": float(dose), "t_hr": float(tm), "ctrl": c_base, "delta": dv})
        paired_map[(cl, drug, dose, tm)] = dv

    pool_obs = obs_m[obs_m["time"] == "3, 6, 12, 24, 48"]
    n_pool = 0
    for hr_str, hr_val in [("3hr", 3.0), ("6hr", 6.0), ("12hr", 12.0), ("24hr", 24.0), ("48hr", 48.0)]:
        dm_tag, tr_tag = f"DMSO_{hr_str}", f"Tram_{hr_str}"
        for cl in pool_obs["cell_line"].unique():
            m_c = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == dm_tag)
            m_t = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == tr_tag)
            if m_c.sum() >= 5 and m_t.sum() >= 10:
                cb = X_m[m_c.values].mean(axis=0)
                dv = (X_m[m_t.values].mean(axis=0) - cb).astype(np.float32)
                samples.append({"src": "MIXseq_Pool", "cl": cl, "drug": "Trametinib", "dose": 0.1, "t_hr": hr_val, "ctrl": cb, "delta": dv})
                n_pool += 1
    del X_m, obs_m

    all_drugs = sorted({s["drug"] for s in samples})
    d2i = {d: i for i, d in enumerate(all_drugs)}
    mol_bank, _ = build_unified_mol_bank(all_drugs)

    def make_feat(drug, dose):
        oh = np.zeros(len(all_drugs), dtype=np.float32)
        oh[d2i[drug]] = 1.0
        return np.concatenate([oh, mol_bank[drug], np.array([np.log1p(max(dose, 0.0))], dtype=np.float32)])

    for s in samples:
        s["mol_feat"] = make_feat(s["drug"], s["dose"])
    d_mol = len(samples[0]["mol_feat"])

    paired_cls = sorted({k[0] for k in paired_map if (k[0], k[1], k[2], "6") in paired_map and (k[0], k[1], k[2], "24") in paired_map})
    rng = np.random.RandomState(SEED)
    rng.shuffle(paired_cls)
    test_cls = set(paired_cls[: max(1, int(0.2 * len(paired_cls)))])

    tr_samples = [s for s in samples if s["cl"] not in test_cls]
    ev_24_all = [s for s in samples if s["src"] == "MIXseq" and s["cl"] in test_cls and s["t_hr"] == 24.0]
    ev_6_pair, ev_24_pair = [], []
    for (cl, drug, dose, tm), dv in paired_map.items():
        if tm == "6" and cl in test_cls and (cl, drug, dose, "24") in paired_map:
            ev_6_pair.append({"ctrl": ctrl_6[cl], "mol_feat": make_feat(drug, float(dose)), "delta": dv})
            ev_24_pair.append({"ctrl": ctrl_6[cl], "mol_feat": make_feat(drug, float(dose)), "delta": paired_map[(cl, drug, dose, "24")]})

    print(f"[MIXseq Clean] 总样本={len(samples)} (含 3-48h 池={n_pool}) | 训练={len(tr_samples)} | 盲测22系全13药24h={len(ev_24_all)} | 配对={len(ev_6_pair)}")

    svd = TruncatedSVD(n_components=48, random_state=SEED)
    svd.fit(np.vstack([s["delta"] for s in tr_samples]))
    U_prog = svd.components_.astype(np.float32)

    bipartite = build_gene_pathway_bipartite_tensors(genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)
    A_prog_prior = (U_prog @ A_signed @ U_prog.T).astype(np.float32)

    time_buckets = {}
    for s in tr_samples:
        time_buckets.setdefault(s["t_hr"], []).append(s)
    gpu_buckets = {
        thr: (
            torch.tensor(np.vstack([r["ctrl"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r["mol_feat"] for r in lst]), device=DEVICE),
            torch.tensor(np.vstack([r["delta"] for r in lst]), device=DEVICE),
        )
        for thr, lst in time_buckets.items()
    }

    torch.manual_seed(SEED)
    model = FastSlowOmniPathODE(n_genes, d_mol, U_prog, A_prog_prior).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180)
    step_cfg = {3.0: 1, 6.0: 2, 12.0: 3, 24.0: 4, 48.0: 6}

    model.train()
    for ep in range(180):
        opt.zero_grad()
        total_loss = 0.0
        for thr, (xc, xm, yd) in gpu_buckets.items():
            pred = model.integrate(xc, xm, t_start=0.0, t_end=thr / 24.0, n_steps=step_cfg[thr])[-1][1]
            total_loss = total_loss + F.mse_loss(pred, yd)
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

        p6_e33b = model.integrate(Xc_e6, Xm_e6, 0.0, 0.25, 2)[-1][1].cpu().numpy()
        p24_e33b_p = model.integrate(Xc_e6, Xm_e6, 0.0, 1.0, 4)[-1][1].cpu().numpy()

        Xc_e24a = torch.tensor(np.vstack([r["ctrl"] for r in ev_24_all]), device=DEVICE)
        Xm_e24a = torch.tensor(np.vstack([r["mol_feat"] for r in ev_24_all]), device=DEVICE)
        Y_e24a = np.vstack([r["delta"] for r in ev_24_all])
        p24_e33b_all = model.integrate(Xc_e24a, Xm_e24a, 0.0, 1.0, 4)[-1][1].cpu().numpy()
        traj_48 = model.integrate(Xc_e24a, Xm_e24a, 0.0, 2.0, 8)

    rows = []
    if os.path.exists(EXP32B_CSV):
        for _, r in pd.read_csv(EXP32B_CSV).iterrows():
            rows.append(dict(r))

    rows.append(eval_delta_metrics(Y_e6, p6_e33b, "Exp33b_MIXseqCleanFastSlow_0h->6h(pair)"))
    rows.append(eval_delta_metrics(Y_e24p, p24_e33b_p, "Exp33b_MIXseqCleanFastSlow_0h->24h(pair)"))
    rows.append(eval_delta_metrics(Y_e24a, p24_e33b_all, "Exp33b_MIXseqCleanFastSlow_0h->24h(All13Drugs)"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Exp33b vs Exp32b 未见22细胞系盲测对比 (总耗时 {time.time()-t0:.1f}s) ===")
    print(out_df.to_string(index=False))

    mem = pd.read_parquet(os.path.join(PRIORS_DIR, "gene_pathway_membership.parquet"))
    hm = mem[mem["source_db"] == "MSigDB_Hallmark_2020"].copy()
    hm["pw_low"] = hm["pathway_name"].str.lower()
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    focus_keywords = ["tnf-alpha", "p53 pathway", "e2f targets", "g2-m checkpoint", "apoptosis", "glycolysis"]
    pw_rows = []
    for t_norm, pred_mat, _, _ in traj_48:
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
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
        for pw, grp_df in pw_df.groupby("pathway"):
            ax.plot(grp_df["time_hours"], grp_df["signed_activity"], marker="o", linewidth=2.0, label=pw)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time Post-Perturbation (Hours, 0h -> 48h)", fontsize=10)
        ax.set_ylabel("Predicted Pathway Activity (Signed Delta)", fontsize=10)
        ax.set_title("Exp33b: Clean MIXseq Fast-Slow ODE Pathway Cascade (0h -> 48h)", fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8.5, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=180)
        print(f"Saved cascade plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
