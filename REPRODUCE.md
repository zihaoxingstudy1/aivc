# Reproducing every reported number

This file maps each manuscript item to the script that produces it and to the output file that is shipped as Source Data. Run all commands from the repository root.

**Data and path conventions.** Scripts read paths from constants near the top of each file. Point them at your local data root once; nothing else needs editing. Primary data are not redistributed — see `DATA.md`. Outputs are written to `test_results/` (plots and predictions) and `source_data/` (the CSVs cited by the manuscript).

**Seeds.** Where a result is stochastic, the five seeds are 42, 101, 202, 303 and 404, passed explicitly on the command line or set in the run block.

---

## Fig. 2 and Table 1 — cross-lineage blind benchmark (n = 2,016 RPE1 perturbations)

| Output | Script |
| :--- | :--- |
| `source_data/Fig1_CrossLineage_SOTA_UQ/sota_benchmark_k562_to_rpe1.csv` | `sota_bench_part1_prepare_and_baselines.py` → `sota_bench_part2_aivc_and_fm.py` → `sota_bench_part2b_align_aivc_v11.py` → `sota_bench_part3_gears_scgpt_scvi.py` → `sota_bench_part3b_scvi_vae.py` → `sota_bench_part4_evaluate_and_plot.py` |
| `sota_benchmark_k562_to_rpe1_stats.csv` (95% CI, paired Wilcoxon, win rate) | `sota_bench_part4_evaluate_and_plot.py` |
| `sota_benchmark_rpe1_calibrated_wmse_pds.csv` | `eval_calibrated_wmse_pds.py` |
| `exp31_lineage_invariant_gnn_results.csv` | `exp31_lineage_invariant_gnn_proteome.py` |
| `exp34_uq_calibration_summary.csv`, `exp34_uq_per_perturbation.csv` | `exp34_uq_calibration_on_exp31.py` |
| `exp44_effect_size_stratified_rpe1.csv`, `exp44_5seed_and_permutation_audit.csv` | `exp44_rigorous_falsification_audit.py` |
| `exp48_exact_sign_ablation_rpe1.csv` | `exp48_exact_sign_ablation_rpe1.py` |
| `exp49_rpe1_component_ablation.csv`, `exp49_rpe1_paired_bootstrap.csv` | `exp49_rpe1_component_ablation_and_matched_baselines.py` |
| `exp49_calibration_results.csv`, `exp49_frozen_params.csv`, `exp49_blend_curve.csv` | `exp49_lolo_oof_calibration.py` |
| Figure panels | `make_nature_fig1_and_stats.py` → `test_results/nature_fig1_sota_multipanel.png` |

Note on file naming: `nature_fig1_*` is the file that appears as **Fig. 2** in the manuscript, because Fig. 1 is the architecture schematic (`nature_fig0_architecture_schematic.png`). The off-by-one is historical; do not renumber the files without updating the manuscript image links.

---

## Table 3b — component and prior ablations

All rows of the ablation table come from `exp49_rpe1_component_ablation_and_matched_baselines.py` and `exp48_exact_sign_ablation_rpe1.py`; the masked-kinase rows come from `exp35_masked_kinase_phospho_and_kalman.py`.

| Ablation row | Script |
| :--- | :--- |
| Prior removed / sign removed / edge subset | `exp48_exact_sign_ablation_rpe1.py` |
| Control-state encoder, teacher embeddings, co-expression, ESM-2, all-inputs-zero | `exp49_rpe1_component_ablation_and_matched_baselines.py` |
| Post-hoc calibration (global scale, ridge blend, per-gene) | `exp49_lolo_oof_calibration.py` |
| Prior on masked kinase (TCGA-OV, CPTAC-LUAD) | `exp35_masked_kinase_phospho_and_kalman.py` |
| Cross-modal pretraining, same-modality pretraining | `exp38_tahoe_drug_temporal_pretrain.py`, `exp36_tahoe100m_two_stage_pretrain.py` |
| Zero-fill ablation | `exp33b_mixseq_fast_slow_clean.py` |
| Overexpression + Joung equal-data audit | `exp46_fair_equal_data_norman_overexpr.py` |
| 18-day anchored ODE vs persistence | `exp41b_schiebinger_anchored_ode.py` |

---

## Fig. 3 and Table 2 — overexpression and double-gene epistasis (Norman2019)

| Output | Script |
| :--- | :--- |
| `exp37c_overexpression_results.csv` | `exp37c_overexpression_cis_omnipath.py` |
| `sota_benchmark_norman_overexpr.csv`, `sota_benchmark_norman_combo.csv` | `sota_bench_norman_part1_fm_and_aivc.py` → `sota_bench_norman_part2_scgpt_gears.py` |
| `exp39_norman_combo_epistasis_results.csv` | `exp39_norman_combo_epistasis_benchmark.py` |
| `exp46_fair_equal_data_norman_overexpr.csv` | `exp46_fair_equal_data_norman_overexpr.py` |
| `exp47_combo_additive_anchored_baselines.csv` | `exp47_combo_additive_anchored_baselines.py` |
| `exp50_combo_component_ablation.csv`, `exp50_combo_paired_bootstrap.csv` | `exp50_combo_component_ablation.py` |
| `exp11_multigene_extrapolation.csv` | `exp11_multigene_extrapolation.py` |
| `exp12_unified_driver_passenger_ranking.csv` | `exp12_unified_driver_passenger.py` |
| Figure panels | `make_nature_fig2_fig3_multipanel.py` → `test_results/nature_fig2_overexpr_and_epistasis.png` |

---

## Fig. 4 and Fig. 5 — pharmacology, temporal dynamics, chemical scaling

| Output | Script |
| :--- | :--- |
| `exp32b_temporal_world_model_results.csv`, `exp32b_temporal_pathway_cascade.csv` | `exp32b_mixseq_joint_temporal_world_model.py` |
| `exp33b_mixseq_fast_slow_results.csv` | `exp33b_mixseq_fast_slow_clean.py` |
| `exp35_masked_kinase_and_kalman_results.csv` | `exp35_masked_kinase_phospho_and_kalman.py` |
| `exp38_tahoe_drug_temporal_results.csv` | `exp38_tahoe_drug_temporal_pretrain.py` |
| `exp40b_leave_drug_out_multikernel_results.csv` | `exp40b_tahoe_multikernel_leave_drug_out.py` |
| `exp41_schiebinger_18day_ode_results.csv`, `exp41b_schiebinger_anchored_ode_results.csv` | `exp41_schiebinger_18day_ode.py`, `exp41b_schiebinger_anchored_ode.py` |
| `exp42_lincs_l1000_mega_scaling_results.csv` | `exp42_lincs_l1000_mega_scaling_law.py` |
| Figure panels | `make_nature_fig2_fig3_multipanel.py` → `test_results/nature_fig3_drug_temporal_scaling.png`; `make_nature_fig2_to_fig5_complete.py` → `test_results/nature_fig4_temporal_and_lincs_scaling.png` |

---

## Fig. 5 and Fig. 6 — five-omics, pan-cancer pharmacology, safety

| Output | Script |
| :--- | :--- |
| `exp27b_local_gated_results.csv` | `exp27b_local_gated_5omics.py` |
| `exp28_learned_regulator_weights.csv`, `exp28_sparse_importance_results.csv` | `exp28_sparse_attention_importance.py` |
| `exp45_*`, `exp45b_*` (80-pair audit, cross-disease targets) | `exp45_target_discovery_3tier_validation.py`, `exp45b_target_discovery_full_audit.py` |
| `aivc_flagship_safety_report.csv` | `aivc_flagship.py` |
| `exp9_pan_cancer_gdsc_ic50.csv` | `exp9_pan_cancer_gdsc_ic50.py` |
| `exp10_drug_combo_synergy.csv` | `exp10_drug_combo_synergy.py` |
| `exp14_unified_gene2drug_transfer.csv` | `exp14_unified_gene_drug_foundation.py` |
| `exp17_prism_and_crispr_lethality.csv` | `exp17_prism_repurposing_and_crispr_lethality.py` |
| `exp19_intrinsic_vs_tme_ablation.csv` | `exp19_clinical_intrinsic_vs_tme_ablation.py` |
| `exp20_wetlab_bulk_consistency.csv` | `exp20_post_pert_matrix_and_wetlab_bulk.py` |
| `exp21_single_drug_gtex_toxicity.csv` | `exp21_flagship_gtex_organ_and_combo_toxicity.py` |
| `exp23_resistant_patient_golden_combos.csv` | `exp23_resistant_patient_golden_combo_discovery.py` |
| Figure panels | `make_nature_fig2_to_fig5_complete.py` → `test_results/nature_fig5_clinical_and_safety.png` |

---

## Extended negative-transfer analyses

| Output | Script |
| :--- | :--- |
| `exp29_cross_lineage_rpe1_results.csv` | `exp29_cross_lineage_lowrank_gate.py` |
| `exp33_expanded_temporal_results.csv` | `exp33_expanded_fast_slow_temporal_world_model.py` |
| `exp36_tahoe_pretrain_results.csv` | `exp36_tahoe100m_two_stage_pretrain.py` |

---

## Supplementary figures

| Item | Script |
| :--- | :--- |
| Supplementary Figs. 1–12 (domain panels) | `make_nature_domain_figures.py` |
| Supplementary Fig. 1 (UQ risk–coverage) | `exp34_uq_calibration_on_exp31.py` |
| Supplementary Fig. 2 (local gating) | `exp27b_local_gated_5omics.py` |
| Supplementary Fig. 4 (L1 importance) | `exp28_sparse_attention_importance.py` |

**Resolved as schematic.** Supplementary Figs. 11–12 (CEBPE+SPI1 and CBL+CNN1 mechanism diagrams) have no generation script and no Source Data table because they are hand-assembled illustrations, not computed panels. They are now labelled *schematic* in the manuscript and the Supplementary information section, and no quantitative claim rests on them. If you later want them reproducible, add the drawing script and its input table rather than upgrading the caption.

---

## Rebuilding the Source Data index

```bash
python test_python_scripts/organize_source_data.py
```

This regenerates `source_data/` from `test_results/` and rewrites `SOURCE_DATA_MANIFEST.md`. It does not recompute anything; run the experiment scripts first.

---

## Verification protocol used for this paper

1. Rebuild the prior atlas and check the directed edge counts (2,838 activation, 1,006 inhibition). A mismatch means the OmniPath snapshot differs from the one used here.
2. Re-run `exp48_exact_sign_ablation_rpe1.py` with seed 42 and confirm the signed arm returns Top-20 *r* = 0.5841. If it does not, stop: the environment differs and no downstream comparison is valid.
3. Run the calibration audit (`exp49_lolo_oof_calibration.py`) and confirm that no arm improves every metric. This is the check that the reported numbers are not a calibration artefact.
4. Re-run the equal-data audit (`exp46_fair_equal_data_norman_overexpr.py`) and confirm the sign pattern: four baselines decrease, AIVC increases.