# -*- coding: utf-8 -*-
"""
omni_fusion.py 测试脚本：
在 Norman 2019 与 Shifrut 2018 (原代 T 细胞) 上评测门控低秩四教师融合架构 (AIVC-OmniFusion)
并与之前的线性基线、单教师、直接拼接版进行打分比较。
"""

import os
import sys
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from omni_fusion import train_and_eval_omni_fusion
from gf_distill import extract_geneformer_gene_embeddings, extract_geneformer_cell_teacher_embeddings
from bfc import build_gene_coexp_prior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


if __name__ == "__main__":
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")

    summaries = []
    for ds_name, h5_path, ens_col in [
        ("Norman2019", r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad", "ensemble_id"),
        ("Shifrut2018", r"D:\data\AIVC\data\Shifrut2018_hvg2000_split.h5ad", "ensembl_id"),
    ]:
        print(f"\n================ {ds_name} OmniFusion 测试 ================")
        adata = sc.read_h5ad(h5_path)
        all_non_ctrl = sorted(set(adata.obs.loc[adata.obs["perturbation"] != "control", "perturbation"].astype(str)))
        vocab = sorted({t.strip() for p in all_non_ctrl for t in p.split("_") if t.strip()})
        var_ens = {g: str(adata.var[ens_col].iloc[i]) for i, g in enumerate(adata.var_names)} if ens_col in adata.var.columns else None

        gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
        scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
        scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)
        coexp_dict = build_gene_coexp_prior(adata[adata.obs["ood_split"] == "train"].X, list(adata.var_names), prior_dim=64)

        teacher_ctx = extract_geneformer_cell_teacher_embeddings(
            adata, gf_dir, tok_pkl, gn_pkl, ensembl_col=ens_col,
            max_seq_len=512, batch_size=32, max_cells_per_group=16,
            pert_col="perturbation", split_col="ood_split", target_split="train",
        )

        teachers = {
            "Coexp": coexp_dict,
            "GF316M": gf_dict,
            "scGPT": scgpt_dict,
            "scFoundation": scf_dict,
            "ESM2": esm2_dict,
        }

        s_df, _ = train_and_eval_omni_fusion(
            adata_or_path=adata,
            teacher_gene_dicts=teachers,
            teacher_cell_embs=teacher_ctx,
            pert_col="perturbation",
            control_label="control",
            sep_char="_",
            split_col="ood_split",
            bottleneck_dim=64,
            hidden_dim=192,
            dropout=0.25,
            epochs=100,
            lr=6e-4,
            weight_decay=5e-3,
            lambda_distill=0.4,
            lambda_mse=25.0,
            model_name="AIVC-OmniFusion(5-Source)",
            save_ckpt_path=rf"D:\data\AIVC\test_results\{ds_name.lower()}_omnifusion.pt",
            per_pert_csv_path=rf"D:\data\AIVC\test_results\{ds_name.lower()}_omnifusion_per_pert.csv",
        )
        s_df["dataset"] = ds_name
        summaries.append(s_df)

    out_df = pd.concat(summaries, ignore_index=True)
    out_df.to_csv(r"D:\data\AIVC\test_results\omnifusion_summary.csv", index=False, encoding="utf-8")
    print("\n=== AIVC-OmniFusion (门控5源融合) 总结 ===")
    print(out_df[["dataset", "model", "pearson_delta_all_mean", "pearson_delta_top20_mean", "dir_acc_top20_mean", "mse_all_mean"]].to_string(index=False))
