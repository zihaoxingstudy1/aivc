# -*- coding: utf-8 -*-
"""
bfc.py 测试脚本：在 Norman 2019 OOD 基准上训练并评测原创模型 BioFlowCell
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from bfc import train_and_eval_bioflowcell


if __name__ == "__main__":
    split_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad"
    ckpt_path = r"D:\data\AIVC\test_results\bioflowcell_norman2019.pt"
    per_pert_csv = r"D:\data\AIVC\test_results\norman2019_bioflowcell_per_pert.csv"
    summary_csv = r"D:\data\AIVC\test_results\norman2019_bioflowcell_summary.csv"

    summary_df, per_pert_df = train_and_eval_bioflowcell(
        adata_or_path=split_h5ad,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        context_col="",
        dose_col="",
        train_split="train",
        val_split="val",
        test_split="test",
        prior_dim=64,
        hidden_dim=256,
        n_blocks=3,
        dropout=0.15,
        ridge_alpha=10.0,
        lr=8e-4,
        weight_decay=1e-4,
        epochs=60,
        cells_per_group_batch=24,
        groups_per_step=8,
        lambda_cfm=0.3,
        lambda_cos=0.8,
        lambda_huber=1.0,
        ode_steps=8,
        seed=42,
        model_name="BioFlowCell-v1",
        top_k_str="20;50",
        save_ckpt_path=ckpt_path,
        per_pert_csv_path=per_pert_csv,
        summary_csv_path=summary_csv,
    )
    print("bfc_test.py 测试完成！")

# 中文注释：在 Norman 2019 未见组合扰动基准上训练并评测 BioFlowCell-v1 模型。
