# -*- coding: utf-8 -*-
"""
BioFlowCell-v3 (Control-Centered Delta-Space Flow) 跨细胞系泛化验证脚本
验证消除跨细胞系绝对表达量偏移 (Domain Shift) 后在 Replogle 与 sci-Plex3 上的表现。
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from bfc import train_and_eval_bioflowcell


if __name__ == "__main__":
    # 1. Replogle K562 -> RPE1
    print(">>> Running BioFlowCell-v3 on Replogle (K562 -> RPE1)...")
    s_rep, _ = train_and_eval_bioflowcell(
        adata_or_path=r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad",
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        context_col="cell_line",
        epochs=15,
        cells_per_group_batch=16,
        groups_per_step=32,
        lr=2e-4,
        dropout=0.30,
        weight_decay=5e-3,
        lambda_cfm=0.25,
        lambda_cos=0.8,
        lambda_huber=1.0,
        lambda_mse=30.0,
        ode_steps=6,
        center_by_context=True,
        model_name="BioFlowCell-v3(Centered)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\replogle_bfc_v3_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\replogle_bfc_v3_summary.csv",
    )

    # 2. sci-Plex3 A549+K562 -> MCF7
    print("\n>>> Running BioFlowCell-v3 on sci-Plex3 (A549+K562 -> MCF7)...")
    s_sci, _ = train_and_eval_bioflowcell(
        adata_or_path=r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad",
        pert_col="pert_dose",
        control_label="control",
        sep_char="||",
        split_col="ood_split",
        context_col="cell_line",
        dose_col="log10_dose",
        epochs=15,
        cells_per_group_batch=16,
        groups_per_step=24,
        lr=2e-4,
        dropout=0.30,
        weight_decay=5e-3,
        lambda_cfm=0.25,
        lambda_cos=0.8,
        lambda_huber=1.0,
        lambda_mse=30.0,
        ode_steps=6,
        center_by_context=True,
        model_name="BioFlowCell-v3(DrugCentered)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\sciplex3_bfc_v3_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\sciplex3_bfc_v3_summary.csv",
    )
    print("\n=== Cross-Context v3 验证完成 ===")
