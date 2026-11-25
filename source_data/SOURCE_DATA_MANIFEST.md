# AIVC 顶刊画图原始数据总清单 (Source Data Manifest)

> **统一存放根目录**：`D:\data\AIVC\source_data`（保留 `test_results/` 原文件不动，此处按 Figure 分文件夹归档，随时可用于重绘任意高阶图表或随顶刊投稿提交 Source Data）。

| 所属图组文件夹 | 文件名 | 行数 × 列数 | 核心用途说明 |
| :--- | :--- | :---: | :--- |
| `Fig1_CrossLineage_SOTA_UQ` | `sota_benchmark_k562_to_rpe1.csv` | `10 × 7` | 10模型跨谱系RPE1盲测汇总均值 |
| `Fig1_CrossLineage_SOTA_UQ` | `sota_benchmark_k562_to_rpe1_stats.csv` | `10 × 18` | 10模型95% Bootstrap CI + 配对Wilcoxon p值 + 胜率 |
| `Fig1_CrossLineage_SOTA_UQ` | `exp31_lineage_invariant_gnn_results.csv` | `4 × 5` | Exp31 v1.1 与 577激酶全掩码对照表 |
| `Fig1_CrossLineage_SOTA_UQ` | `exp34_uq_calibration_summary.csv` | `4 × 5` | Exp34 5成员深集成UQ校准与风险-覆盖度汇总 |
| `Fig1_CrossLineage_SOTA_UQ` | `exp34_uq_per_perturbation.csv` | `2016 × 7` | Exp34 逐扰动(2,016条)不确定性与真实MSE明细 |
| `Fig2_Overexpr_Epistasis_MultiGene` | `exp37c_overexpression_results.csv` | `4 × 5` | Exp37c 纯基因过表达盲测(Joung->Norman2019) |
| `Fig2_Overexpr_Epistasis_MultiGene` | `sota_benchmark_norman_overexpr.csv` | `8 × 5` | Norman2019 8模型开源SOTA过表达统一排行榜(含scGPT/Geneformer-V2/scFoundation/GEARS) |
| `Fig2_Overexpr_Epistasis_MultiGene` | `exp39_norman_combo_epistasis_results.csv` | `5 × 7` | Exp39 未见双基因组合与上位性残差盲测 |
| `Fig2_Overexpr_Epistasis_MultiGene` | `sota_benchmark_norman_combo.csv` | `9 × 6` | Norman2019 9模型开源SOTA双基因组合与上位性统一排行榜 |
| `Fig2_Overexpr_Epistasis_MultiGene` | `exp11_multigene_extrapolation.csv` | `876 × 11` | Exp11 3-6基因高阶组合零样本外推 |
| `Fig2_Overexpr_Epistasis_MultiGene` | `exp12_unified_driver_passenger_ranking.csv` | `135 × 7` | Exp12 925组多基因Driver vs Passenger归因排名 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp32b_temporal_world_model_results.csv` | `10 × 5` | Exp32b MIXseq 22未见细胞系24h时序世界模型 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp33b_mixseq_fast_slow_results.csv` | `13 × 5` | Exp33b 快慢双尺度灰箱ODE 6h/24h盲测 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp32b_temporal_pathway_cascade.csv` | `54 × 4` | Exp32b 0h-48h 通路级联演化轨迹数据 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp38_tahoe_drug_temporal_results.csv` | `4 × 6` | Exp38 Tahoe-100M同模态药物预训练6h/24h对比 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp40b_leave_drug_out_multikernel_results.csv` | `7 × 7` | Exp40b 76未见新药Leave-Drug-Out与单调Scaling Law |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp42_lincs_l1000_mega_scaling_results.csv` | `8 × 7` | Exp42 LINCS L1000 千药级Leave-Drug-Out零样本盲测与Scaling Law |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp35_masked_kinase_and_kalman_results.csv` | `35 × 9` | Exp35 盲测肿瘤激酶[MASK]磷酸化恢复与6h->24h卡尔曼同化 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp41_schiebinger_18day_ode_results.csv` | `4 × 5` | Exp41 Schiebinger 18天重编程开环ODE外推 |
| `Fig3_Drug_Temporal_Scaling_Phospho` | `exp41b_schiebinger_anchored_ode_results.csv` | `3 × 5` | Exp41b Schiebinger 18天重编程D12状态锚定ODE外推(反超Persistence_Day12) |
| `Fig4_5Omics_Mechanistic_Discovery` | `exp27b_local_gated_results.csv` | `16 × 4` | Exp27b 5-Omics局部图门控LUAD/OV盲测 |
| `Fig4_5Omics_Mechanistic_Discovery` | `exp28_sparse_importance_results.csv` | `8 × 3` | Exp28 L1稀疏注意力重要性与因果调控指标 |
| `Fig4_5Omics_Mechanistic_Discovery` | `exp28_learned_regulator_weights.csv` | `3904 × 3` | Exp28 自动发现的核心癌基因调控边权重 |
| `Fig5_Clinical_PanCancer_Safety` | `exp9_pan_cancer_gdsc_ic50.csv` | `2 × 8` | Exp9 GDSC1+GDSC2 45万对泛癌IC50盲测 |
| `Fig5_Clinical_PanCancer_Safety` | `exp10_drug_combo_synergy.csv` | `4 × 9` | Exp10 DrugComb+Sanger 双药协同预测 |
| `Fig5_Clinical_PanCancer_Safety` | `exp14_unified_gene2drug_transfer.csv` | `4 × 4` | Exp14 基因上位性向双药协同迁移(AUC=0.8389) |
| `Fig5_Clinical_PanCancer_Safety` | `exp17_prism_and_crispr_lethality.csv` | `4 × 8` | Exp17 PRISM老药新用与DepMap CRISPR致死率 |
| `Fig5_Clinical_PanCancer_Safety` | `exp19_intrinsic_vs_tme_ablation.csv` | `51 × 9` | Exp19 17临床队列(5,265例)四臂消融 |
| `Fig5_Clinical_PanCancer_Safety` | `exp20_wetlab_bulk_consistency.csv` | `5 × 10` | Exp20 外部GEO湿实验Bulk RNA-seq一致性验证 |
| `Fig5_Clinical_PanCancer_Safety` | `exp23_resistant_patient_golden_combos.csv` | `1770 × 8` | Exp23 531例耐药乳腺癌患者双药Pareto治疗窗 |
| `Fig5_Clinical_PanCancer_Safety` | `exp21_single_drug_gtex_toxicity.csv` | `18 × 11` | Exp21 GTEx 54器官单药毒性评分 |
| `Fig5_Clinical_PanCancer_Safety` | `aivc_flagship_safety_report.csv` | `6 × 6` | 旗舰模块GTEx 54器官 + EyeGEx 453视网膜联合安全窗 |
| `Extended_NegativeTransfer_Ablations` | `exp33_expanded_temporal_results.csv` | `13 × 5` | Exp33 跨平台缺失517基因零填充负迁移消融 |
| `Extended_NegativeTransfer_Ablations` | `exp36_tahoe_pretrain_results.csv` | `2 × 5` | Exp36 跨模态药物初始化遗传敲低负迁移消融 |
| `Extended_NegativeTransfer_Ablations` | `exp29_cross_lineage_rpe1_results.csv` | `2 × 5` | Exp29 v1.0冻结基线与消融明细 |

> **补充大型逐样本原始预测矩阵（`.npy` / `.npz`）存放位置**：`D:\data\AIVC\test_results\sota_bench_preds\`（包含 10 个模型在 `RPE1` 2,016 个扰动 × 3,603 基因的完整预测矩阵与 `rpe1_ood_bundle.npz`，可用于任意时候重算 p 值、画小提琴图/散点图/基因热图）。
