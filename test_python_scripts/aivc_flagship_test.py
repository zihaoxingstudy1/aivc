# -*- coding: utf-8 -*-
"""
aivc_flagship_test.py
End-to-end verification of func_python_scripts/aivc_flagship.py on real data:
  1. load_flagship_priors + compute_omnipath_causal_footprints
  2. compute_ensemble_uq_scores
  3. kalman_assimilate_state
  4. assess_multi_organ_and_retina_safety (GTEx v8 54 organs + EyeGEx 453 human retinas)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from aivc_flagship import (
    load_flagship_priors,
    compute_omnipath_causal_footprints,
    compute_ensemble_uq_scores,
    kalman_assimilate_state,
    assess_multi_organ_and_retina_safety,
)


def main():
    t0 = time.time()
    print("=== AIVC Flagship Integration Test ===")
    bundle = np.load(r"D:\data\AIVC\test_results\sota_bench_preds\rpe1_ood_bundle.npz", allow_pickle=True)
    target_genes = [str(g) for g in bundle["target_genes"]]

    # 1. Priors + OmniPath causal footprints
    bip = load_flagship_priors(target_genes, r"D:\data\AIVC\data\priors", min_genes_per_pathway=3, include_biogrid=False)
    u_dummy = np.eye(64, len(target_genes), dtype=np.float32)
    f_causal = compute_omnipath_causal_footprints(bip["A_signed"], u_dummy, hop2_weight=0.5)
    assert f_causal.shape == (len(target_genes), 64)
    print(f"  [1/4] Priors & Causal Footprints OK: {f_causal.shape}, kinases={int(bip['kinase_like_mask'].sum())}")

    # 2. Ensemble UQ on real AIVC v1.1 prediction
    p_v11 = np.load(r"D:\data\AIVC\test_results\sota_bench_preds\pred_AIVC_v1.1.npy")
    p_ens = np.load(r"D:\data\AIVC\test_results\sota_bench_preds\pred_AIVC_v1.1_Ensemble.npy")
    uq = compute_ensemble_uq_scores([p_v11, p_ens], ood_distances=np.linalg.norm(p_ens - p_v11, axis=1))
    assert uq["mean_pred"].shape == p_v11.shape
    print(f"  [2/4] Ensemble UQ OK: mean_std={uq['ensemble_std'].mean():.5f}")

    # 3. Kalman state assimilation
    assimilated = kalman_assimilate_state(p_ens[:10], p_v11[:10], kalman_gain=0.1)
    assert assimilated.shape == (10, len(target_genes))
    print(f"  [3/4] Kalman Assimilation OK: shape={assimilated.shape}")

    # 4. GTEx v8 (54 organs) + EyeGEx (453 retinas) safety guard
    safety_df = assess_multi_organ_and_retina_safety(
        target_genes_str="GLP1R;EGFR;ERBB2;BRAF;MAP2K1;RPE65",
        gtex_gct_gz_path=r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz",
        eyegex_parquet_path=r"D:\data\AIVC\data\EyeGEx_GSE115828_retina_median.parquet",
        tpm_risk_threshold=1.0,
        retina_cpm_risk_threshold=1.0,
    )
    out_csv = r"D:\data\AIVC\test_results\aivc_flagship_safety_report.csv"
    safety_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("  [4/4] Multi-Organ + EyeGEx Retina Safety Report:")
    print(safety_df.to_string(index=False))
    print(f"\nAll AIVC Flagship integration checks passed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
