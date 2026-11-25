# -*- coding: utf-8 -*-
"""
quickstart_verify.py
Self-contained verification script that runs immediately after `git clone`
and `pip install -e .` without requiring external primary datasets (`data/`).

Verifies:
  1. `aivc` public package import & all exported symbols
  2. Core functional operators (causal footprints, ensemble UQ, Kalman assimilation)
  3. RDKit chemical ADMET/druglikeness scoring on EGFR inhibitors (Gefitinib, Osimertinib)
  4. Bundled Source Data integrity (Table 1 RPE1 benchmark, Table 2 combos, ablations)
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
FUNC_DIR = os.path.join(REPO_ROOT, "func_python_scripts")
if FUNC_DIR not in sys.path:
    sys.path.insert(0, FUNC_DIR)


def main():
    print("=== AIVC Out-of-the-Box Reproducibility & Smoke Verification ===")

    # 1. Package import
    import aivc
    assert aivc.__version__ == "1.1.0"
    assert len(aivc.__all__) >= 11
    print(f"[1/4] Package import OK: aivc v{aivc.__version__} ({len(aivc.__all__)} exported symbols)")

    # 2. Core algorithmic operators
    rng = np.random.default_rng(42)
    n_genes, n_pert = 64, 8
    A_signed = np.zeros((n_genes, n_genes), dtype=np.float32)
    A_signed[0, 1] = 1.0
    A_signed[1, 2] = -1.0
    u = np.eye(n_pert, n_genes, dtype=np.float32)
    f_causal = aivc.compute_omnipath_causal_footprints(A_signed, u, hop2_weight=0.5)
    assert f_causal.shape == (n_genes, n_pert)

    preds = [rng.normal(0, 0.1, size=(10, n_genes)).astype(np.float32) for _ in range(3)]
    uq = aivc.compute_ensemble_uq_scores(preds)
    assert uq["mean_pred"].shape == (10, n_genes)
    assimilated = aivc.kalman_assimilate_state(preds[0], preds[1], kalman_gain=0.1)
    assert assimilated.shape == (10, n_genes)
    print(f"[2/4] Core operators OK: footprints={f_causal.shape}, UQ mean_std={uq['ensemble_std'].mean():.4f}")

    # 3. Chemical ADMET interface (EGFR inhibitors)
    smiles_map = {
        "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "Osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
    }
    admet_df = aivc.compute_admet_druglikeness(list(smiles_map.values()), names=list(smiles_map.keys()))
    assert len(admet_df) == 2
    print(f"[3/4] Chemical ADMET interface OK: scored {len(admet_df)} EGFR inhibitors")

    # 4. Bundled Source Data integrity check
    sd_rpe1 = os.path.join(REPO_ROOT, "source_data", "Fig1_CrossLineage_SOTA_UQ", "sota_benchmark_rpe1_calibrated_wmse_pds.csv")
    sd_combo = os.path.join(REPO_ROOT, "source_data", "Fig2_Overexpr_Epistasis_MultiGene", "exp47_combo_additive_anchored_baselines.csv")
    sd_ablation = os.path.join(REPO_ROOT, "source_data", "Fig1_CrossLineage_SOTA_UQ", "exp49_rpe1_component_ablation.csv")
    for p in (sd_rpe1, sd_combo, sd_ablation):
        assert os.path.isfile(p), f"Missing Source Data file: {p}"
    df_rpe1 = pd.read_csv(sd_rpe1)
    df_combo = pd.read_csv(sd_combo)
    df_abl = pd.read_csv(sd_ablation)
    print(
        f"[4/4] Bundled Source Data OK: RPE1 models={len(df_rpe1)}, "
        f"Norman combo baselines={len(df_combo)}, RPE1 ablation arms={len(df_abl)}"
    )
    print("\nALL CHECKS PASSED. To reproduce full training/benchmarks on primary datasets, see DATA.md and REPRODUCE.md.")


if __name__ == "__main__":
    main()
