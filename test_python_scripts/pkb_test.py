# -*- coding: utf-8 -*-
import os
import sys
import json

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from pkb import build_full_prior_atlas

OUT_DIR = r"D:\data\AIVC\data\priors"
BIOGRID_PATH = r"D:\data\regulation\BIOGRID-ALL-4.4.211.tab3.txt"
CHECK_H5AD = r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad"

if __name__ == "__main__":
    summary = build_full_prior_atlas(
        out_dir=OUT_DIR,
        biogrid_path=BIOGRID_PATH,
        library_names="KEGG_2026;Reactome_Pathways_2024;WikiPathways_2024_Human;MSigDB_Hallmark_2020;TRRUST_Transcription_Factors_2019",
        min_genes_per_pathway=5,
        max_genes_per_pathway=1000,
        min_shared_genes=3,
        min_jaccard=0.02,
        check_h5ad_path=CHECK_H5AD,
    )
    print("\n=== 全量人类通路与调控先验图谱汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
