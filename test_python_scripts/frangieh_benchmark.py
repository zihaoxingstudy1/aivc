# -*- coding: utf-8 -*-
"""
Frangieh 2021 (黑色素瘤 + IFN-γ + TIL 免疫共培养) 基准构建与 AIVC-OmniFusion 评测
任务：在 Control 与 IFNγ 微环境下训练，Zero-shot 预测真实 TIL 免疫细胞共培养 (Co-culture) 下的基因敲除响应。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from dlsp import load_and_standardize_perturb_data
from ops import split_ood_perturbations
from lbb import run_linear_baseline_benchmark
from omni_fusion import train_and_eval_omni_fusion
from gf_distill import extract_geneformer_gene_embeddings, extract_geneformer_cell_teacher_embeddings
from bfc import build_gene_coexp_prior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_scfoundation_gene_embeddings


if __name__ == "__main__":
    raw_h5 = r"D:\data\AIVC\data\FrangiehIzar2021_RNA.h5ad"
    std_h5 = r"D:\data\AIVC\data\Frangieh2021_hvg2000.h5ad"
    split_h5 = r"D:\data\AIVC\data\Frangieh2021_hvg2000_split.h5ad"

    if not os.path.exists(split_h5):
        adata, _ = load_and_standardize_perturb_data(
            h5ad_path=raw_h5,
            pert_col="perturbation",
            control_label="control",
            sep_char="_",
            quality_col="",
            min_cells_per_gene=50,
            min_cells_per_pert=30,
            target_sum=10000.0,
            n_top_genes=2000,
            keep_pert_genes=True,
            save_h5ad_path=std_h5,
            summary_csv_path=r"D:\data\AIVC\test_results\frangieh2021_pert_summary.csv",
        )

        # 跨微环境 OOD 划分: Train = Control + IFNγ (部分做 Val), Test = Co-culture (TIL 免疫共培养)
        adata_split, _ = split_ood_perturbations(
            adata_or_path=adata,
            pert_col="perturbation",
            control_label="control",
            sep_char="_",
            split_mode="unseen_context",
            context_col="perturbation_2",
            val_contexts="",
            test_contexts="Co-culture",
            seed=42,
            save_h5ad_path=split_h5,
            split_summary_csv=r"D:\data\AIVC\test_results\frangieh2021_ood_split_summary.csv",
        )
    else:
        adata_split = sc.read_h5ad(split_h5)

    # 1. 线性基线评测
    s_base, _ = run_linear_baseline_benchmark(
        adata_or_path=adata_split,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        train_split="train",
        eval_split="test",
        ridge_alpha=10.0,
        top_k_str="20;50",
        per_pert_csv_path=r"D:\data\AIVC\test_results\frangieh2021_baseline_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\frangieh2021_baseline_summary.csv",
        plot_png_path=r"D:\data\AIVC\test_results\frangieh2021_baseline_benchmark.png",
    )

    # 2. AIVC-OmniFusion (多教师融合) 评测
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"

    all_non_ctrl = sorted(set(adata_split.obs.loc[adata_split.obs["perturbation"] != "control", "perturbation"].astype(str)))
    vocab = sorted({t.strip() for p in all_non_ctrl for t in p.split("_") if t.strip()})
    var_ens = {g: str(adata_split.var["ensembl_id"].iloc[i]) for i, g in enumerate(adata_split.var_names)}

    gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)
    coexp_dict = build_gene_coexp_prior(adata_split[adata_split.obs["ood_split"] == "train"].X, list(adata_split.var_names), prior_dim=64)

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict}
    s_omni, _ = train_and_eval_omni_fusion(
        adata_or_path=adata_split,
        teacher_gene_dicts=teachers,
        teacher_cell_embs=None,
        pert_col="perturbation",
        control_label="control",
        sep_char="_",
        split_col="ood_split",
        bottleneck_dim=64,
        hidden_dim=192,
        dropout=0.25,
        epochs=60,
        lr=4e-4,
        weight_decay=5e-3,
        lambda_mse=40.0,
        model_name="AIVC-OmniFusion(Melanoma-TIL)",
        per_pert_csv_path=r"D:\data\AIVC\test_results\frangieh2021_omnifusion_per_pert.csv",
        summary_csv_path=r"D:\data\AIVC\test_results\frangieh2021_omnifusion_summary.csv",
    )

    all_df = pd.concat([s_base, s_omni], ignore_index=True)
    all_df.to_csv(r"D:\data\AIVC\test_results\frangieh2021_all_models_summary.csv", index=False, encoding="utf-8")
    print("\n=== Frangieh 2021 (黑色素瘤 TIL 共培养) 全模型汇总 ===")
    print(all_df[["model", "pearson_delta_all_mean", "pearson_delta_top20_mean", "dir_acc_top20_mean", "mse_all_mean"]].to_string(index=False))
