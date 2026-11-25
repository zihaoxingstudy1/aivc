# -*- coding: utf-8 -*-
"""
Exp18: 端到端下游生信机制解释与发表级可视化验证
结合 Exp12b 识别的强协同组合 (CEBPE_SPI1 与 CBL_CNN1) 及 AIVC-OmniFusion 预测的真实转录组 Delta，
自动执行 GO Biological Process & KEGG 通路富集分析、气泡图绘制，以及 靶点->基因->通路 三层调控网络图绘制。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import _to_dense_array
from bio_pathway_viz import run_go_kegg_enrichment, plot_target_pathway_network


def run_exp18():
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad")
    gene_names = list(adata.var_names)
    cm = (adata.obs["perturbation"].astype(str) == "control").values
    mu_ctrl = _to_dense_array(adata.X[cm]).mean(axis=0).astype(np.float32)

    mech_df = pd.read_csv(r"D:\data\AIVC\test_results\exp12b_norman_mechanism.csv").set_index("combo")

    for combo in ["CEBPE_SPI1", "CBL_CNN1"]:
        pm = (adata.obs["perturbation"].astype(str) == combo).values
        if pm.sum() == 0:
            continue
        delta = _to_dense_array(adata.X[pm]).mean(axis=0).astype(np.float32) - mu_ctrl

        out_csv = rf"D:\data\AIVC\test_results\exp18_{combo}_go_kegg.csv"
        out_bub = rf"D:\data\AIVC\test_results\exp18_{combo}_enrichment_bubble.png"
        out_net = rf"D:\data\AIVC\test_results\exp18_{combo}_regulatory_network.png"

        enr_df = run_go_kegg_enrichment(
            pred_delta=delta,
            gene_names=gene_names,
            pert_label=combo,
            out_csv=out_csv,
            out_bubble_png=out_bub,
            top_n_genes=80,
        )

        row = mech_df.loc[combo]
        targets = [str(row["driver_gene"]), str(row["passenger_gene"])]
        shares = [float(row["driver_share"]), float(row["passenger_share"])]
        plot_target_pathway_network(
            target_names=targets,
            target_shares=shares,
            pred_delta=delta,
            gene_names=gene_names,
            enrich_df=enr_df,
            pert_label=combo,
            out_net_png=out_net,
            top_pathways=6,
            max_genes_per_pathway=5,
        )


if __name__ == "__main__":
    run_exp18()
