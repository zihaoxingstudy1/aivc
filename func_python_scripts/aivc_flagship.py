# -*- coding: utf-8 -*-
"""
aivc_flagship.py
Project 1 (AIVC Virtual Cell World Model) Unified Flagship Integration Module:
  1. Full Human Pathway & Directed OmniPath Prior Atlas (pkb + pgnn)
  2. Dual-Track 5-Omics & Masked-Kinase Protein/Phospho Completion (Exp28 + Exp35)
  3. Lineage-Invariant Causal Perturbation Predictor + 5-Member Deep Ensemble UQ (Exp31 + Exp34)
  4. Fast-Slow Grey-Box Temporal ODE + Kalman State Assimilation (Exp32b/33b + Exp35)
  5. Multi-Organ (GTEx v8 54 Organs) + Human Retina (EyeGEx GSE115828) Safety Guard
"""

import os
import gzip
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pgnn import build_gene_pathway_bipartite_tensors
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex


def load_flagship_priors(
    target_genes: List[str],
    priors_dir: str,
    min_genes_per_pathway: int = 3,
    include_biogrid: bool = True,
) -> dict:
    """Load gene-pathway bipartite & OmniPath directed signed adjacency tensors."""
    priors_dir = str(priors_dir)
    target_genes = [str(g).strip() for g in target_genes]
    bip = build_gene_pathway_bipartite_tensors(
        target_genes=target_genes,
        priors_dir=priors_dir,
        min_genes_per_pathway=int(min_genes_per_pathway),
        include_biogrid=bool(include_biogrid),
    )
    a_signed = (bip["A_pos"] - bip["A_neg"]).astype(np.float32)
    bip["A_signed"] = a_signed
    return bip


def compute_omnipath_causal_footprints(
    a_signed: np.ndarray,
    u_prog: np.ndarray,
    hop2_weight: float = 0.5,
) -> np.ndarray:
    """Compute 1-2 hop lineage-invariant OmniPath causal perturbation footprints in program space."""
    a_signed = np.asarray(a_signed, dtype=np.float32)
    u_prog = np.asarray(u_prog, dtype=np.float32)
    hop2_weight = float(hop2_weight)
    a2 = a_signed @ a_signed
    return ((a_signed + hop2_weight * a2) @ u_prog.T).astype(np.float32)


def compute_ensemble_uq_scores(
    member_preds: List[np.ndarray],
    ood_distances: Optional[np.ndarray] = None,
    std_weight: float = 0.7,
    ood_weight: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Compute calibrated Deep Ensemble epistemic uncertainty + OOD distance scores."""
    stack = np.stack([np.asarray(p, dtype=np.float32) for p in member_preds], axis=0)
    mean_pred = stack.mean(axis=0).astype(np.float32)
    ens_std = stack.std(axis=0).mean(axis=1).astype(np.float32)

    if ood_distances is not None:
        ood_d = np.asarray(ood_distances, dtype=np.float32)
        s_norm = (ens_std - ens_std.mean()) / (ens_std.std() + 1e-6)
        d_norm = (ood_d - ood_d.mean()) / (ood_d.std() + 1e-6)
        combined = (float(std_weight) * s_norm + float(ood_weight) * d_norm).astype(np.float32)
    else:
        combined = ens_std
    return {
        "mean_pred": mean_pred,
        "ensemble_std": ens_std,
        "combined_uq": combined,
    }


def kalman_assimilate_state(
    pred_state: np.ndarray,
    obs_state: np.ndarray,
    kalman_gain: float = 0.1,
) -> np.ndarray:
    """Assimilate noisy intermediate observation with ODE prior state."""
    k = float(np.clip(kalman_gain, 0.0, 1.0))
    return ((1.0 - k) * np.asarray(pred_state, dtype=np.float32) + k * np.asarray(obs_state, dtype=np.float32)).astype(np.float32)


def assess_multi_organ_and_retina_safety(
    target_genes_str: str,
    gtex_gct_gz_path: str,
    eyegex_parquet_path: str,
    tpm_risk_threshold: float = 1.0,
    retina_cpm_risk_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate target expression safety across GTEx v8 (54 human organs) and EyeGEx (453 human retinas).
    Multi-gene targets are passed as semicolon-separated string (e.g. 'GLP1R;EGFR;ERBB2').
    """
    genes = [g.strip().upper() for g in str(target_genes_str).split(";") if g.strip()]
    if not genes:
        return pd.DataFrame()
    gene_set = set(genes)

    # 1. GTEx v8 54 organs
    gtex_df = pd.read_csv(str(gtex_gct_gz_path), sep="\t", skiprows=2, compression="gzip", low_memory=False)
    gtex_df["sym_up"] = gtex_df["Description"].astype(str).str.strip().str.upper()
    gtex_sub = gtex_df[gtex_df["sym_up"].isin(gene_set)].drop_duplicates("sym_up").set_index("sym_up")
    organ_cols = [c for c in gtex_df.columns if c not in ("Name", "Description", "sym_up")]

    # 2. EyeGEx human retina (453 donors) + Ensembl -> Symbol mapping
    ens_to_sym = load_ensembl_to_sym_from_gtex(str(gtex_gct_gz_path))
    eye_df = pd.read_parquet(str(eyegex_parquet_path))
    eye_gene_col = eye_df.columns[0]
    eye_val_col = [c for c in eye_df.columns if c != eye_gene_col][0]
    eye_df["sym_up"] = [ens_to_sym.get(str(x).split(".")[0].strip(), str(x).strip().upper()).upper() for x in eye_df[eye_gene_col]]
    eye_map = eye_df[eye_df["sym_up"].isin(gene_set)].groupby("sym_up")[eye_val_col].median().to_dict()

    rows = []
    for g in genes:
        retina_val = float(eye_map.get(g, 0.0))
        if g in gtex_sub.index:
            organ_vals = gtex_sub.loc[g, organ_cols].astype(float)
            top_organ = str(organ_vals.idxmax())
            top_tpm = float(organ_vals.max())
            n_organs_above = int((organ_vals >= float(tpm_risk_threshold)).sum())
        else:
            top_organ, top_tpm, n_organs_above = "N/A", 0.0, 0
        risk_flag = (n_organs_above > 0) or (retina_val >= float(retina_cpm_risk_threshold))
        rows.append({
            "gene": g,
            "gtex_top_organ": top_organ,
            "gtex_top_tpm": top_tpm,
            "gtex_organs_above_thresh": n_organs_above,
            "eyegex_retina_median": retina_val,
            "safety_alert": bool(risk_flag),
        })
    return pd.DataFrame(rows)
