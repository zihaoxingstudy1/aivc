# -*- coding: utf-8 -*-
"""
gf_distill.py 测试脚本：在 Norman 2019 OOD 基准上执行 Geneformer-V2-104M_CLcancer 知识蒸馏评测
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from gf_distill import train_and_eval_distill_student


if __name__ == "__main__":
    split_h5ad = r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad"
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-104M_CLcancer"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"

    summary_df, per_pert_df, _ = train_and_eval_distill_student(
        adata_or_path=split_h5ad,
        geneformer_dir=gf_dir,
        token_dict_pkl=tok_pkl,
        gene_name_id_pkl=gn_pkl,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        ensembl_col="ensemble_id",
        ridge_alpha=10.0,
        hidden_dim=256,
        epochs=100,
        lr=8e-4,
        lambda_distill=0.5,
        top_k_str="20;50",
        model_name="DistillCell-GF104M",
        save_ckpt_path=r"D:\data\AIVC\test_results\distillcell_gf_norman2019.pt",
        per_pert_csv_path=r"D:\data\AIVC\test_results\norman2019_distillcell_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\norman2019_distillcell_summary.csv",
    )
    print("gf_distill_test.py 测试全部通过！")

# 中文注释：调用 gf_distill 模块在 Norman 2019 基准上完成 Geneformer-V2-104M 知识蒸馏训练与评测。
