# -*- coding: utf-8 -*-
"""
exp38_tahoe_drug_temporal_pretrain.py
Exp38: Same-Modality Drug Perturbation Pretraining (Tahoe-100M 379 Drugs x 50+ Cell Lines)
       -> Fine-Tuning FastSlowOmniPathODE on MIXseq (3h/6h/12h/24h/48h)
       -> Blind Evaluation on the exact 22 Unseen Cell Lines vs Exp32b / Exp33b.
"""

import os
import sys
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from pgnn import build_gene_pathway_bipartite_tensors
from chem_interface import encode_smiles_multimodal

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp33_expanded_fast_slow_temporal_world_model import (
    FastSlowOmniPathODE, build_unified_mol_bank, eval_delta_metrics
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
TAHOE_META_DIR = r"D:\data\AIVC\data\tahoe_100m\metadata"
TAHOE_PB_DIR = os.path.join(TAHOE_META_DIR, "pseudobulk_differential_expression")
CHEMBERTA_DIR = r"D:\data\AIVC\pretrained\chemberta"

OUT_CSV = r"D:\data\AIVC\test_results\exp38_tahoe_drug_temporal_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp38_tahoe_drug_temporal_plot.png"


def load_tahoe_drug_corpus(mix_genes: list, max_shards: int = 15, max_samples: int = 2500):
    """Load Tahoe-100M pseudobulk drug deltas aligned to MIXseq 2000 HVGs + SMILES features."""
    drug_meta = pd.read_parquet(os.path.join(TAHOE_META_DIR, "drug_metadata.parquet"))
    smi_map = {}
    for d, smi in zip(drug_meta["drug"], drug_meta["canonical_smiles"]):
        if pd.notna(d) and pd.notna(smi) and str(smi).strip():
            smi_map[str(d).strip()] = str(smi).strip()

    g2i = {g.upper(): i for i, g in enumerate(mix_genes)}
    gset = set(g2i.keys())
    shards = sorted(glob.glob(os.path.join(TAHOE_PB_DIR, "train-*.parquet")))[:max_shards]

    vecs = {}
    for sp in shards:
        df = pd.read_parquet(sp, columns=["gene_name", "log2FoldChange", "Cell_Name_Vevo", "drug", "concentration"])
        df["g_up"] = df["gene_name"].astype(str).str.upper()
        df = df[df["g_up"].isin(gset)].dropna(subset=["log2FoldChange"])
        for (cl, drg, conc), grp in df.groupby(["Cell_Name_Vevo", "drug", "concentration"], sort=False):
            drg_s = str(drg).strip()
            if drg_s not in smi_map:
                continue
            key = (str(cl), drg_s, float(conc))
            if key not in vecs:
                if len(vecs) >= max_samples:
                    continue
                vecs[key] = np.zeros(len(mix_genes), dtype=np.float32)
            idxs = [g2i[g] for g in grp["g_up"].values]
            vecs[key][idxs] = np.clip(grp["log2FoldChange"].values.astype(np.float32), -4.0, 4.0)
        if len(vecs) >= max_samples:
            break

    keys = list(vecs.keys())
    Y_tahoe = np.vstack([vecs[k] for k in keys]).astype(np.float32)
    # Match MIXseq delta std (~0.20)
    Y_tahoe = Y_tahoe / (np.std(Y_tahoe) + 1e-6) * 0.20

    uniq_drugs = sorted({k[1] for k in keys})
    morg, cb = encode_smiles_multimodal([smi_map[d] for d in uniq_drugs], chemberta_dir=CHEMBERTA_DIR, device_str="cpu")
    mol_feats = {d: np.hstack([cb[i], morg[i]]).astype(np.float32) for i, d in enumerate(uniq_drugs)}

    F_tahoe = np.vstack([
        np.concatenate([
            mol_feats[k[1]],
            np.array([np.log10(max(k[2], 1e-4) + 1e-4) / 3.0], dtype=np.float32)
        ])
        for k in keys
    ]).astype(np.float32)
    print(f"  [Tahoe-100M Drug Corpus] samples={len(Y_tahoe)}, unique_drugs={len(uniq_drugs)}, feat_dim={F_tahoe.shape[1]}")
    return Y_tahoe, F_tahoe


def main():
    t0 = time.time()
    print(f"=== Exp38: Tahoe-100M Same-Modality Drug Pretraining -> MIXseq Temporal World Model ({DEVICE}) ===")

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
    for hr_str, hr_val in [("3hr", 3.0), ("6hr", 6.0), ("12hr", 12.0), ("24hr", 24.0), ("48hr", 48.0)]:
        dm_tag, tr_tag = f"DMSO_{hr_str}", f"Tram_{hr_str}"
        for cl in pool_obs["cell_line"].unique():
            m_c = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == dm_tag)
            m_t = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == tr_tag)
            if m_c.sum() >= 5 and m_t.sum() >= 10:
                cb = X_m[m_c.values].mean(axis=0)
                dv = (X_m[m_t.values].mean(axis=0) - cb).astype(np.float32)
                samples.append({"src": "MIXseq_Pool", "cl": cl, "drug": "Trametinib", "dose": 0.1, "t_hr": hr_val, "ctrl": cb, "delta": dv})
    del X_m, obs_m

    all_drugs = sorted({s["drug"] for s in samples})
    mol_bank, _ = build_unified_mol_bank(all_drugs)

    # Pure continuous structure+dose feature (zero drug one-hot so Tahoe & MIXseq share exact input space!)
    for s in samples:
        s["mol_feat"] = np.concatenate([
            mol_bank[s["drug"]],
            np.array([np.log10(max(s["dose"], 1e-4) + 1e-4) / 3.0], dtype=np.float32)
        ]).astype(np.float32)
    d_mol = len(samples[0]["mol_feat"])

    paired_cls = sorted({cl for (cl, d, ds, tm) in paired_map if tm == "6" and (cl, d, ds, "24") in paired_map})
    rng = np.random.RandomState(SEED)
    rng.shuffle(paired_cls)
    test_cls = set(paired_cls[: max(1, int(0.2 * len(paired_cls)))])

    tr_samples = [s for s in samples if s["cl"] not in test_cls]
    ev_24_all = [s for s in samples if s["src"] == "MIXseq" and s["cl"] in test_cls and s["t_hr"] == 24.0]
    ev_6_pair = []
    for (cl, drug, dose, tm), dv in paired_map.items():
        if tm == "6" and cl in test_cls and (cl, drug, dose, "24") in paired_map:
            ev_6_pair.append({"ctrl": ctrl_6[cl], "mol_feat": samples[0]["mol_feat"] * 0 + np.concatenate([mol_bank[drug], np.array([np.log10(max(float(dose), 1e-4) + 1e-4) / 3.0], dtype=np.float32)]), "delta": dv})

    print(f"[MIXseq] train={len(tr_samples)} | test_24h_all13={len(ev_24_all)} | test_6h_pair={len(ev_6_pair)} | unseen_cls={len(test_cls)}")

    # Load Tahoe-100M shared drug corpus
    Y_tahoe, F_tahoe = load_tahoe_drug_corpus(genes, max_shards=15, max_samples=2500)

    # Joint SVD program basis (48 dims)
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
    step_cfg = {3.0: 1, 6.0: 2, 12.0: 3, 24.0: 4, 48.0: 6}

    Xc_24 = torch.tensor(np.vstack([r["ctrl"] for r in ev_24_all]), device=DEVICE)
    Xm_24 = torch.tensor(np.vstack([r["mol_feat"] for r in ev_24_all]), device=DEVICE)
    Y_24 = np.vstack([r["delta"] for r in ev_24_all])

    Xc_6 = torch.tensor(np.vstack([r["ctrl"] for r in ev_6_pair]), device=DEVICE)
    Xm_6 = torch.tensor(np.vstack([r["mol_feat"] for r in ev_6_pair]), device=DEVICE)
    Y_6 = np.vstack([r["delta"] for r in ev_6_pair])

    mean_ctrl = np.mean([s["ctrl"] for s in tr_samples], axis=0).astype(np.float32)
    Xc_tahoe_dummy = np.repeat(mean_ctrl[None, :], len(Y_tahoe), axis=0)

    results = []
    for use_pretrain in [False, True]:
        tag = "Exp38_TahoeDrugPretrained" if use_pretrain else "Exp38_NoPretrain_StructOnly"
        torch.manual_seed(SEED)
        model = FastSlowOmniPathODE(n_genes, d_mol, U_prog, A_prog_prior).to(DEVICE)

        if use_pretrain:
            opt_p = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
            dl_p = DataLoader(TensorDataset(torch.tensor(Xc_tahoe_dummy), torch.tensor(F_tahoe), torch.tensor(Y_tahoe)), batch_size=128, shuffle=True)
            model.train()
            for ep in range(25):
                for bc, bm, by in dl_p:
                    bc, bm, by = bc.to(DEVICE), bm.to(DEVICE), by.to(DEVICE)
                    opt_p.zero_grad()
                    pred = model.integrate(bc, bm, t_start=0.0, t_end=1.0, n_steps=4)[-1][1]
                    F.mse_loss(pred, by).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt_p.step()
            print(f"  [{tag}] Stage 1 Tahoe-100M drug pretraining (25 epochs) complete")

        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180)
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
            p24 = model.integrate(Xc_24, Xm_24, 0.0, 1.0, 4)[-1][1].cpu().numpy()
            p6 = model.integrate(Xc_6, Xm_6, 0.0, 0.25, 2)[-1][1].cpu().numpy()

        m24 = eval_delta_metrics(Y_24, p24, f"{tag}_24h_All13")
        m6 = eval_delta_metrics(Y_6, p6, f"{tag}_6h_Pair")
        for horizon, m in [("24h_All13Drugs", m24), ("6h_Paired", m6)]:
            print(f"  [{tag:28s} | {horizon:14s}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
            results.append({
                "model": tag,
                "horizon": horizon,
                "mean_pearson": m["mean_pearson"],
                "top20_pearson": m["top20_pearson"],
                "dir_acc_top20": m["dir_acc_top20"],
                "mse": m["mse"],
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    for ax, hor in zip(axes, ["6h_Paired", "24h_All13Drugs"]):
        sub = res_df[res_df["horizon"] == hor]
        x = np.arange(len(sub))
        w = 0.28
        ax.bar(x - w, sub["mean_pearson"], width=w, label="Mean Pearson r", color="#3498db", edgecolor="black", linewidth=0.5)
        ax.bar(x, sub["top20_pearson"], width=w, label="Top20 Pearson r", color="#2ecc71", edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sub["dir_acc_top20"], width=w, label="DirAcc Top20", color="#e67e22", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["No Pretrain\n(StructOnly)", "Tahoe-100M\nPretrained"], fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"MIXseq 22 Unseen Cell Lines ({hor})", fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Exp38: Same-Modality Tahoe-100M Drug Pretraining on Temporal ODE", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp38 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
