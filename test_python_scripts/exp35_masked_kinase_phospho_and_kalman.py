# -*- coding: utf-8 -*-
"""
exp35_masked_kinase_phospho_and_kalman.py
Exp35 (No-Download Validation):
  Part A: Quantitative Masked-Kinase Phosphorylation Recovery on Blind Unseen Human Tumors
          (TCGA-RPPA & CPTAC-Phospho LUAD & OV) using Exp28 + OmniPath Causal Footprint.
  Part B: Obs6h -> 24h Kalman-Style State Assimilation on MIXseq 22 Unseen Cell Lines
          using Exp33b Fast-Slow Grey-Box ODE.
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
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from pgnn import build_gene_pathway_bipartite_tensors
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp27b_local_gated_5omics import (
    LocalGated5OmicsPathwayGNN, row_zscore, RPPA_TO_GENE,
    TCGA_RNA_DIR, TCGA_RPPA_PATH, TCGA_MAF_DIR, CPTAC_DIR, CPTAC_MUT_DIR,
    HOTSPOT_PATH, OUT_CKPT, UNSEEN_CANCERS, CPTAC_MUT_PREFIX,
)
from exp33_expanded_fast_slow_temporal_world_model import (
    FastSlowOmniPathODE, build_unified_mol_bank, eval_delta_metrics,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp35_masked_kinase_and_kalman_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp35_masked_kinase_and_kalman_plot.png"


def build_omnipath_causal_features(X_rna: np.ndarray, A_signed: np.ndarray) -> np.ndarray:
    """Build signed 1-hop + 2-hop OmniPath transcriptional footprint features."""
    f1_out = X_rna @ A_signed.T
    f1_in = X_rna @ A_signed
    f2_out = f1_out @ A_signed.T
    return np.concatenate([f1_out, f1_in, f2_out], axis=1).astype(np.float32)


def median_r_subset(y_pred: np.ndarray, y_true: np.ndarray, col_indices: list) -> tuple:
    rs = []
    mses = []
    for j in col_indices:
        yt = y_true[:, j]
        yp = y_pred[:, j]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            r = pearsonr(yp, yt)[0]
            if np.isfinite(r):
                rs.append(r)
        mses.append(float(np.mean((yp - yt) ** 2)))
    return (float(np.median(rs)) if rs else 0.0), (float(np.mean(mses)) if mses else 0.0), len(rs)


def run_part_a_masked_kinase_phospho():
    print("\n--- Part A: Masked-Kinase Phosphorylation Recovery (TCGA-RPPA & CPTAC-Phospho) ---")
    ckpt = torch.load(OUT_CKPT, map_location=DEVICE, weights_only=False)
    top_rna = ckpt["top_rna"]
    top_prot = ckpt["top_prot"]
    top_phos = ckpt["top_phos"]
    top_mir = ckpt["top_mir"]
    rppa_abs = ckpt["rppa_abs"]
    driver_genes = ckpt["driver_genes"]
    driver_set = set(driver_genes)
    drv_pos = {g: i for i, g in enumerate(driver_genes)}
    rppa_target_genes = [RPPA_TO_GENE.get(ab, ab.split("_")[0].lstrip("X").upper()) for ab in rppa_abs]

    net = LocalGated5OmicsPathwayGNN(
        len(top_rna), len(driver_genes), len(top_mir),
        ckpt["rppa_cis"], ckpt["prot_cis"], ckpt["phos_cis"],
        ckpt["A_tf"], ckpt["A_gp"], ckpt["M_rppa"], ckpt["M_prot"], ckpt["M_phos"]
    ).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    bip = build_gene_pathway_bipartite_tensors(top_rna, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    kinase_mask_rna = np.asarray(bip["kinase_like_mask"], dtype=bool)
    kinase_genes = {top_rna[i].upper() for i in np.where(kinase_mask_rna)[0]}
    print(f"  Kinase hubs in top_rna: {len(kinase_genes)} / {len(top_rna)}")

    # Identify phospho subsets
    rppa_phos_idx = [j for j, ab in enumerate(rppa_abs) if "_p" in ab]
    rppa_masked_kin_idx = [j for j in rppa_phos_idx if rppa_target_genes[j].upper() in kinase_genes]
    cp_all_phos_idx = list(range(len(top_phos)))
    cp_masked_kin_idx = [j for j, g in enumerate(top_phos) if g.upper() in kinase_genes]
    print(f"  TCGA-RPPA phospho={len(rppa_phos_idx)}, Masked-Kinase phospho={len(rppa_masked_kin_idx)}")
    print(f"  CPTAC-MS phospho={len(cp_all_phos_idx)}, Masked-Kinase phospho={len(cp_masked_kin_idx)}")

    # Load TCGA
    rppa = pd.read_csv(TCGA_RPPA_PATH, sep="\t", index_col=0)
    rppa = rppa[rppa.isna().mean(axis=1) < 0.30]
    rppa_sids = set(rppa.columns)
    sid_to_mut = {}
    for mf in sorted(glob.glob(os.path.join(TCGA_MAF_DIR, "TCGA_*_mutect2_maf"))):
        df_m = pd.read_csv(mf, sep="\t", comment="#", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"], low_memory=False)
        df_m["Hugo_Symbol"] = df_m["Hugo_Symbol"].astype(str).str.strip().str.upper()
        df_m = df_m[(df_m["Variant_Classification"] != "Silent") & (df_m["Hugo_Symbol"].isin(driver_set))]
        for sym, bc in zip(df_m["Hugo_Symbol"].values, df_m["Tumor_Sample_Barcode"].astype(str).values):
            s15 = bc[:15]
            if s15 in rppa_sids:
                if s15 not in sid_to_mut:
                    sid_to_mut[s15] = np.zeros(len(driver_genes), dtype=np.float32)
                sid_to_mut[s15][drv_pos[sym]] = 1.0

    tx_l, tm_l, ty_l, tc_l = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(TCGA_RNA_DIR, "*_mrna_tpm.csv"))):
        ctype = os.path.basename(fp).replace("TCGA-", "").replace("_mrna_tpm.csv", "")
        df_r = pd.read_csv(fp, low_memory=False)
        df_r["gene"] = df_r["gene"].astype(str).str.strip().str.upper()
        df_r = df_r[df_r["gene"].isin(top_rna)].drop_duplicates("gene").set_index("gene")
        col_to_s15 = {c: str(c)[:15] for c in df_r.columns if str(c)[:15] in rppa_sids}
        if not col_to_s15:
            continue
        seen = {}
        for oc, s15 in col_to_s15.items():
            if s15 not in seen:
                seen[s15] = oc
        s15s = sorted(seen.keys())
        tx_l.append(row_zscore(np.log2(np.clip(df_r[[seen[s] for s in s15s]].reindex(top_rna).fillna(0.0).values.T, 0.0, None) + 1.0)))
        tm_l.append(np.stack([sid_to_mut.get(s, np.zeros(len(driver_genes), dtype=np.float32)) for s in s15s]))
        ty_l.append(np.nan_to_num(rppa[s15s].values.T.astype(np.float32), nan=0.0))
        tc_l.extend([ctype] * len(s15s))
    X_tcga_r, X_tcga_mut, Y_tcga_rppa, C_tcga = np.vstack(tx_l), np.vstack(tm_l), np.vstack(ty_l), np.array(tc_l)

    # Load CPTAC
    ens_to_sym = load_ensembl_to_sym_from_gtex(os.path.join(DATA_DIR, "GTEx_v8_gene_median_tpm.gct.gz"))
    cptac_cancers = sorted(CPTAC_MUT_PREFIX.keys())
    sm, cp_raw = None, {}
    for c in cptac_cancers:
        r = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_transcriptomics_bcm_processed_data.csv"), index_col=0)
        p = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_proteomics_bcm_processed_data.csv"), index_col=0)
        ph = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_phosphoproteomics_bcm_processed_data.csv"), index_col=0)
        r.index = [str(i).strip().upper() for i in r.index]; r = r.groupby(r.index, sort=False).mean()
        p.index = [str(i).strip().upper() for i in p.index]; p = p.groupby(p.index, sort=False).mean()
        ph.index = [str(i).split(":")[0].strip().upper() for i in ph.index]; ph = ph.groupby(ph.index, sort=False).mean()
        mir_p = os.path.join(CPTAC_DIR, f"{c}_miRNA_bcm_processed_data.csv")
        m_df = pd.read_csv(mir_p, index_col=0) if os.path.exists(mir_p) else None
        if m_df is not None:
            m_df.index = [str(i).strip().lower() for i in m_df.index]; m_df = m_df.groupby(m_df.index, sort=False).mean()
            sm = set(m_df.index) if sm is None else (sm & set(m_df.index))
        mut_p = os.path.join(CPTAC_MUT_DIR, f"{CPTAC_MUT_PREFIX[c]}_somatic_mutation_gene_level_binary.txt")
        mut_df = pd.read_csv(mut_p, sep="\t", index_col=0)
        mut_df["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in mut_df.index]
        mut_df = mut_df[mut_df["__sym__"].isin(driver_set)].groupby("__sym__", sort=False).max()
        cp_raw[c] = (r, m_df, p, ph, mut_df)

    cx_l, cmut_l, cmir_l, chas_mir_l, cph_l, cc_l = [], [], [], [], [], []
    for c in cptac_cancers:
        r, m_df, p, ph, mut_df = cp_raw[c]
        sids = sorted(set(r.columns) & set(p.columns) & set(ph.columns))
        cx_l.append(row_zscore(r[sids].reindex(top_rna).fillna(0.0).T.values))
        cph_l.append(row_zscore(ph[sids].reindex(top_phos).fillna(0.0).T.values))
        cmut_l.append(mut_df.reindex(index=driver_genes, columns=sids).fillna(0.0).T.values.astype(np.float32))
        if m_df is not None:
            mir_mat = np.zeros((len(sids), len(top_mir)), dtype=np.float32)
            has_m = np.array([s in m_df.columns for s in sids], dtype=bool)
            vp = np.where(has_m)[0]
            if len(vp):
                mir_mat[vp] = row_zscore(m_df[[sids[i] for i in vp]].reindex(top_mir).fillna(0.0).T.values)
            cmir_l.append(mir_mat); chas_mir_l.append(has_m)
        else:
            cmir_l.append(np.zeros((len(sids), len(top_mir)), dtype=np.float32)); chas_mir_l.append(np.zeros(len(sids), dtype=bool))
        cc_l.extend([c.upper()] * len(sids))
    X_cp_r, X_cp_mut, X_cp_mir, has_cp_mir, Y_cp_phos, C_cp = (
        np.vstack(cx_l), np.vstack(cmut_l), np.vstack(cmir_l), np.concatenate(chas_mir_l), np.vstack(cph_l), np.array(cc_l)
    )

    # Fit OmniPath Causal Residual Adapter strictly on SEEN cancers (zero LUAD/OV leakage)
    tr_t = ~np.isin(C_tcga, list(UNSEEN_CANCERS))
    tr_c = ~np.isin(C_cp, list(UNSEEN_CANCERS))

    X_tcga_r_masked = X_tcga_r.copy()
    X_tcga_r_masked[:, kinase_mask_rna] = 0.0
    X_cp_r_masked = X_cp_r.copy()
    X_cp_r_masked[:, kinase_mask_rna] = 0.0

    with torch.no_grad():
        pred_tr_t_masked = net.forward_rppa(
            torch.tensor(X_tcga_r_masked[tr_t], device=DEVICE),
            torch.tensor(X_tcga_mut[tr_t], device=DEVICE), None
        ).cpu().numpy()
        pred_tr_c_masked = net.forward_phos(
            torch.tensor(X_cp_r_masked[tr_c], device=DEVICE),
            torch.tensor(X_cp_mut[tr_c], device=DEVICE),
            torch.tensor(X_cp_mir[tr_c], device=DEVICE)
        ).cpu().numpy()

    A_signed = (bip["A_pos"] - bip["A_neg"]).astype(np.float32)
    causal_tcga_tr = build_omnipath_causal_features(X_tcga_r_masked[tr_t], A_signed)
    causal_cp_tr = build_omnipath_causal_features(X_cp_r_masked[tr_c], A_signed)
    ridge_tcga = Ridge(alpha=500.0).fit(causal_tcga_tr, Y_tcga_rppa[tr_t] - pred_tr_t_masked)
    ridge_cp = Ridge(alpha=500.0).fit(causal_cp_tr, Y_cp_phos[tr_c] - pred_tr_c_masked)

    rows = []
    rppa_cis = np.array(ckpt["rppa_cis"])
    phos_cis = np.array(ckpt["phos_cis"])

    for c in sorted(UNSEEN_CANCERS):
        mt = (C_tcga == c)
        xt_r = torch.tensor(X_tcga_r[mt], device=DEVICE)
        xt_rm = torch.tensor(X_tcga_r_masked[mt], device=DEVICE)
        xt_m = torch.tensor(X_tcga_mut[mt], device=DEVICE)
        with torch.no_grad():
            yp_unmasked = net.forward_rppa(xt_r, xt_m, None).cpu().numpy()
            yp_masked = net.forward_rppa(xt_rm, xt_m, None).cpu().numpy()
        yp_causal = yp_masked + 0.5 * ridge_tcga.predict(build_omnipath_causal_features(X_tcga_r_masked[mt], A_signed))
        # Naive cis-mRNA baseline under kinase masking
        yp_cis_masked = np.zeros_like(yp_masked)
        for j, idx_r in enumerate(rppa_cis):
            if idx_r >= 0:
                yp_cis_masked[:, j] = X_tcga_r_masked[mt, idx_r]

        for mname, ypred in [
            ("Cis_mRNA_MaskedKinase", yp_cis_masked),
            ("Exp28_5Omics_Unmasked", yp_unmasked),
            ("Exp28_5Omics_MaskedKinase", yp_masked),
            ("Exp28+OmniPathCausal_MaskedKinase", yp_causal),
        ]:
            r_all, mse_all, n_all = median_r_subset(ypred, Y_tcga_rppa[mt], rppa_phos_idx)
            r_kin, mse_kin, n_kin = median_r_subset(ypred, Y_tcga_rppa[mt], rppa_masked_kin_idx)
            rows.append({"part": "A_MaskedKinase", "cohort": f"TCGA-{c}", "subset": "All_Phospho", "model": mname, "median_r": r_all, "mse": mse_all, "n_targets": n_all})
            rows.append({"part": "A_MaskedKinase", "cohort": f"TCGA-{c}", "subset": "Masked_Kinase_Phospho", "model": mname, "median_r": r_kin, "mse": mse_kin, "n_targets": n_kin})
            print(f"  [TCGA-{c:4s} | {mname:33s}] AllPhospho r={r_all:.4f} | MaskedKinase r={r_kin:.4f}")

        mc = (C_cp == c)
        xc_r = torch.tensor(X_cp_r[mc], device=DEVICE)
        xc_rm = torch.tensor(X_cp_r_masked[mc], device=DEVICE)
        xc_m = torch.tensor(X_cp_mut[mc], device=DEVICE)
        xc_mi = torch.tensor(X_cp_mir[mc], device=DEVICE) if has_cp_mir[mc].any() else None
        with torch.no_grad():
            yc_unmasked = net.forward_phos(xc_r, xc_m, xc_mi).cpu().numpy()
            yc_masked = net.forward_phos(xc_rm, xc_m, xc_mi).cpu().numpy()
        yc_causal = yc_masked + 0.5 * ridge_cp.predict(build_omnipath_causal_features(X_cp_r_masked[mc], A_signed))
        yc_cis_masked = np.zeros_like(yc_masked)
        for j, idx_r in enumerate(phos_cis):
            if idx_r >= 0:
                yc_cis_masked[:, j] = X_cp_r_masked[mc, idx_r]

        for mname, ypred in [
            ("Cis_mRNA_MaskedKinase", yc_cis_masked),
            ("Exp28_5Omics_Unmasked", yc_unmasked),
            ("Exp28_5Omics_MaskedKinase", yc_masked),
            ("Exp28+OmniPathCausal_MaskedKinase", yc_causal),
        ]:
            r_all, mse_all, n_all = median_r_subset(ypred, Y_cp_phos[mc], cp_all_phos_idx)
            r_kin, mse_kin, n_kin = median_r_subset(ypred, Y_cp_phos[mc], cp_masked_kin_idx)
            rows.append({"part": "A_MaskedKinase", "cohort": f"CPTAC-{c}", "subset": "All_Phospho", "model": mname, "median_r": r_all, "mse": mse_all, "n_targets": n_all})
            rows.append({"part": "A_MaskedKinase", "cohort": f"CPTAC-{c}", "subset": "Masked_Kinase_Phospho", "model": mname, "median_r": r_kin, "mse": mse_kin, "n_targets": n_kin})
            print(f"  [CPTAC-{c:3s} | {mname:33s}] AllPhospho r={r_all:.4f} | MaskedKinase r={r_kin:.4f}")

    return rows


def run_part_b_kalman_assimilation():
    print("\n--- Part B: Obs6h -> 24h Kalman-Style State Assimilation (MIXseq 22 Unseen Cell Lines) ---")
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
        samples.append({"cl": cl, "drug": drug, "dose": float(dose), "t_hr": float(tm), "ctrl": c_base, "delta": dv})
        paired_map[(cl, drug, dose, tm)] = (c_base, dv)

    pool_obs = obs_m[obs_m["time"] == "3, 6, 12, 24, 48"]
    for hr_str, hr_val in [("3hr", 3.0), ("6hr", 6.0), ("12hr", 12.0), ("24hr", 24.0), ("48hr", 48.0)]:
        dm_tag, tr_tag = f"DMSO_{hr_str}", f"Tram_{hr_str}"
        for cl in pool_obs["cell_line"].unique():
            m_c = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == dm_tag)
            m_t = (obs_m["time"] == "3, 6, 12, 24, 48") & (obs_m["cell_line"] == cl) & (obs_m["hash_tag"] == tr_tag)
            if m_c.sum() >= 5 and m_t.sum() >= 10:
                cb = X_m[m_c.values].mean(axis=0)
                dv = (X_m[m_t.values].mean(axis=0) - cb).astype(np.float32)
                samples.append({"cl": cl, "drug": "Trametinib", "dose": 0.1, "t_hr": hr_val, "ctrl": cb, "delta": dv})
    del X_m, obs_m

    all_drugs = sorted({s["drug"] for s in samples})
    d2i = {d: i for i, d in enumerate(all_drugs)}
    mol_bank, d_mol_raw = build_unified_mol_bank(all_drugs)

    def make_feat(drug, dose):
        oh = np.zeros(len(all_drugs), dtype=np.float32)
        oh[d2i[drug]] = 1.0
        mol = mol_bank[drug]
        return np.concatenate([oh, mol, np.array([np.log10(max(dose, 1e-4) + 1e-4) / 3.0], dtype=np.float32)])

    for s in samples:
        s["mol_feat"] = make_feat(s["drug"], s["dose"])
    d_mol = len(samples[0]["mol_feat"])

    paired_cls = sorted({cl for (cl, d, ds, tm) in paired_map if tm == "6" and (cl, d, ds, "24") in paired_map})
    rng = np.random.RandomState(SEED)
    rng.shuffle(paired_cls)
    test_cls = set(paired_cls[: max(1, int(0.2 * len(paired_cls)))])

    tr_samples = [s for s in samples if s["cl"] not in test_cls]
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
    tr_pairs, te_pairs = [], []
    for (cl, drug, dose, tm), (cb24, dv24) in paired_map.items():
        if tm != "24" or (cl, drug, dose, "6") not in paired_map:
            continue
        cb6, dv6 = paired_map[(cl, drug, dose, "6")]
        item = {"cl": cl, "drug": drug, "dose": float(dose), "ctrl": cb6, "mol_feat": make_feat(drug, float(dose)), "d6": dv6, "d24": dv24}
        if cl not in test_cls:
            tr_pairs.append(item)
        else:
            te_pairs.append(item)

    def predict_kalman(pairs, K_gain: float):
        xc = torch.tensor(np.vstack([p["ctrl"] for p in pairs]), device=DEVICE)
        xm = torch.tensor(np.vstack([p["mol_feat"] for p in pairs]), device=DEVICE)
        d6_obs = torch.tensor(np.vstack([p["d6"] for p in pairs]), device=DEVICE)
        with torch.no_grad():
            traj_6 = model.integrate(xc, xm, t_start=0.0, t_end=0.25, n_steps=2)
            _, _, zf6, zs6 = traj_6[-1]
            z_obs6 = d6_obs @ model.U_prog.t()
            err_z = z_obs6 - (zf6 + zs6)
            zf_assim = zf6 + 0.5 * K_gain * err_z
            zs_assim = zs6 + 0.5 * K_gain * err_z
            p24_kal = model.integrate(xc, xm, t_start=0.25, t_end=1.0, n_steps=3, zf0=zf_assim, zs0=zs_assim)[-1][1].cpu().numpy()
            p24_open = model.integrate(xc, xm, t_start=0.0, t_end=1.0, n_steps=4)[-1][1].cpu().numpy()
        return p24_kal, p24_open, np.vstack([p["d24"] for p in pairs])

    best_k, best_tr_r = 0.5, -1.0
    for k_cand in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_kal, _, y_t = predict_kalman(tr_pairs, k_cand)
        m = eval_delta_metrics(y_t, p_kal, "cand")
        if m["mean_pearson"] > best_tr_r:
            best_tr_r = m["mean_pearson"]
            best_k = k_cand

    p_kal_te, p_open_te, y_te = predict_kalman(te_pairs, best_k)
    p_persist_te = np.vstack([p["d6"] for p in te_pairs])
    print(f"  Paired (6h->24h) Test Cases on {len(test_cls)} Unseen Cell Lines: {len(te_pairs)} | Train-selected Kalman Gain K={best_k}")

    rows = []
    for name, pred in [
        ("Persistence_Obs6h", p_persist_te),
        ("OpenLoop_ODE_0h->24h", p_open_te),
        (f"Kalman_ODE_0h+Obs6h->24h (K={best_k})", p_kal_te),
    ]:
        m = eval_delta_metrics(y_te, pred, name)
        print(f"  [{name:34s}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.6f}")
        rows.append({
            "part": "B_Kalman6h24h",
            "cohort": f"MIXseq_Unseen{len(test_cls)}",
            "subset": f"Paired_6h_24h (n={len(te_pairs)})",
            "model": name,
            "median_r": m["mean_pearson"],
            "top20_pearson": m["top20_pearson"],
            "dir_acc_top20": m["dir_acc_top20"],
            "mse": m["mse"],
            "n_targets": len(te_pairs),
        })
    return rows


def main():
    t0 = time.time()
    print(f"=== Exp35: Masked-Kinase Phospho Recovery & Kalman 6h->24h Assimilation ({DEVICE}) ===")
    rows_a = run_part_a_masked_kinase_phospho()
    rows_b = run_part_b_kalman_assimilation()
    df = pd.DataFrame(rows_a + rows_b)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=180)
    # Panel 1: Masked-Kinase Phospho Recovery
    sub_a = df[(df["part"] == "A_MaskedKinase") & (df["subset"] == "Masked_Kinase_Phospho")]
    piv_a = sub_a.pivot(index="cohort", columns="model", values="median_r")
    col_order = ["Cis_mRNA_MaskedKinase", "Exp28_5Omics_MaskedKinase", "Exp28+OmniPathCausal_MaskedKinase", "Exp28_5Omics_Unmasked"]
    piv_a = piv_a[[c for c in col_order if c in piv_a.columns]]
    piv_a.plot(kind="bar", ax=axes[0], color=["#95a5a6", "#e67e22", "#2ecc71", "#3498db"], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Part A: Masked-Kinase Phosphorylation Recovery\n(Blind LUAD & OV Tumors)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Median Cross-Patient Pearson (r)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=7.5, loc="lower right")

    # Panel 2: Kalman 6h -> 24h Assimilation
    sub_b = df[df["part"] == "B_Kalman6h24h"]
    x_pos = np.arange(len(sub_b))
    w = 0.35
    axes[1].bar(x_pos - w / 2, sub_b["median_r"], width=w, label="Mean Pearson r", color="#2980b9", edgecolor="black", linewidth=0.5)
    axes[1].bar(x_pos + w / 2, sub_b["top20_pearson"], width=w, label="Top20 Pearson r", color="#27ae60", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([m.replace(" ", "\n") for m in sub_b["model"]], fontsize=8)
    axes[1].set_ylabel("Correlation on 24h Delta (Unseen 22 Cell Lines)")
    axes[1].set_title("Part B: Obs6h -> 24h Kalman State Assimilation\n(MIXseq Paired 6h & 24h)", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp35 results -> {OUT_CSV} & {OUT_PNG} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
