# -*- coding: utf-8 -*-
"""
pkb.py - 全量人类通路与有向调控先验图谱构建模块 (Prior Knowledge Builder)
将 KEGG_2026 / Reactome_Pathways_2024 / WikiPathways_2024_Human / MSigDB_Hallmark_2020 /
TRRUST / OmniPath 有向正负调控 / BioGRID 物理互作 统一整理为“基因-通路二部图 + 通路交错网”底座。
"""

import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
import gseapy as gp


def _normalize_gene_symbol(sym: str) -> str:
    """标准化人类基因符号为大写去空格字符串。"""
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    if s in {"", "-", "NAN", "NONE", "NA"}:
        return ""
    return s


def fetch_gseapy_pathway_libraries(
    library_names: str,
    out_dir: str,
    organism: str = "Human",
    min_genes_per_pathway: int = 5,
    max_genes_per_pathway: int = 1000,
) -> pd.DataFrame:
    """
    批量拉取并清洗全量人类通路库，生成标准化基因-通路隶属表。
    参数:
        library_names: 分号分隔的 gseapy 库名，例如 "KEGG_2026;Reactome_Pathways_2024;..."
        out_dir: 输出目录
        organism: 物种名称，默认 Human
        min_genes_per_pathway: 通路最小基因数过滤阈值
        max_genes_per_pathway: 通路最大基因数过滤阈值（剔除过于宽泛的顶层泛集）
    """
    os.makedirs(out_dir, exist_ok=True)
    libs = [x.strip() for x in str(library_names).split(";") if x.strip()]
    records = []

    for lib in libs:
        print(f"[PKB] 正在拉取通路库: {lib} (organism={organism}) ...")
        raw_dict = gp.get_library(name=lib, organism=organism)
        kept = 0
        for pw_name, genes in raw_dict.items():
            clean_genes = sorted({_normalize_gene_symbol(g) for g in genes} - {""})
            if len(clean_genes) < int(min_genes_per_pathway) or len(clean_genes) > int(max_genes_per_pathway):
                continue
            kept += 1
            pw_id = f"{lib}::{pw_name}"
            for g in clean_genes:
                records.append((g, pw_id, str(pw_name), lib))
        print(f"  -> {lib} 原始通路数={len(raw_dict)}, 过滤后保留={kept}")

    df = pd.DataFrame(records, columns=["gene_symbol", "pathway_id", "pathway_name", "source_db"])
    out_path = os.path.join(out_dir, "gene_pathway_membership.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[PKB] 已保存基因-通路隶属表: {out_path} (边数={len(df)}, 唯一基因={df['gene_symbol'].nunique()}, 唯一通路={df['pathway_id'].nunique()})")
    return df


def fetch_omnipath_directed_edges(
    out_dir: str,
    datasets: str = "omnipath,pathwayextra,kinaseextra,ligrecextra,dorothea",
    organism_id: int = 9606,
    min_curation_effort: int = 1,
    timeout_sec: int = 90,
) -> pd.DataFrame:
    """
    从 OmniPath 公开 REST API 下载全量人类有向且带激活/抑制符号的信号与转录调控边。
    """
    os.makedirs(out_dir, exist_ok=True)
    url = "https://omnipathdb.org/interactions"
    params = {
        "datasets": str(datasets),
        "organisms": int(organism_id),
        "genesymbols": 1,
        "directed": 1,
        "fields": "sources,references,curation_effort,type,dorothea_level",
        "format": "tsv",
    }
    print(f"[PKB] 正在从 OmniPath 下载人类有向调控网络 ({datasets}) ...")
    resp = requests.get(url, params=params, timeout=int(timeout_sec))
    resp.raise_for_status()

    raw_tsv = os.path.join(out_dir, "omnipath_raw_human.tsv")
    with open(raw_tsv, "w", encoding="utf-8") as f:
        f.write(resp.text)

    df = pd.read_csv(raw_tsv, sep="\t", low_memory=False)
    df["src"] = df["source_genesymbol"].map(_normalize_gene_symbol)
    df["dst"] = df["target_genesymbol"].map(_normalize_gene_symbol)
    df = df[(df["src"] != "") & (df["dst"] != "") & (df["src"] != df["dst"])].copy()

    if "curation_effort" in df.columns:
        df["curation_effort"] = pd.to_numeric(df["curation_effort"], errors="coerce").fillna(0).astype(int)
        df = df[df["curation_effort"] >= int(min_curation_effort)].copy()

    stim = pd.to_numeric(df.get("is_stimulation", 0), errors="coerce").fillna(0).astype(int)
    inh = pd.to_numeric(df.get("is_inhibition", 0), errors="coerce").fillna(0).astype(int)
    df["sign"] = np.where((stim == 1) & (inh == 0), 1, np.where((inh == 1) & (stim == 0), -1, 0))
    df["is_directed"] = pd.to_numeric(df.get("is_directed", 1), errors="coerce").fillna(1).astype(int)

    keep_cols = ["src", "dst", "sign", "is_directed", "curation_effort", "type", "sources"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["src", "dst", "sign"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, "directed_signaling_edges.parquet")
    df.to_parquet(out_path, index=False)
    n_pos = int((df["sign"] == 1).sum())
    n_neg = int((df["sign"] == -1).sum())
    print(f"[PKB] 已保存有向调控边: {out_path} (总有向边={len(df)}, 激活={n_pos}, 抑制={n_neg}, 覆盖基因={len(set(df['src']) | set(df['dst']))})")
    return df


def parse_biogrid_human_ppi(
    biogrid_path: str,
    out_dir: str,
    organism_id: int = 9606,
    system_type_filter: str = "physical",
    chunksize: int = 400000,
) -> pd.DataFrame:
    """
    流式解析本地 BioGRID tab3 大文件，提取人类物理互作蛋白对（作为局部复合体与交错补充骨架）。
    """
    os.makedirs(out_dir, exist_ok=True)
    usecols = [
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "Experimental System Type",
        "Organism ID Interactor A",
        "Organism ID Interactor B",
    ]
    pairs = set()
    print(f"[PKB] 正在流式解析本地 BioGRID: {biogrid_path} ...")
    for chunk in pd.read_csv(biogrid_path, sep="\t", usecols=usecols, chunksize=int(chunksize), low_memory=False):
        org_a = pd.to_numeric(chunk["Organism ID Interactor A"], errors="coerce")
        org_b = pd.to_numeric(chunk["Organism ID Interactor B"], errors="coerce")
        mask = (org_a == int(organism_id)) & (org_b == int(organism_id))
        if system_type_filter:
            mask &= (chunk["Experimental System Type"].astype(str).str.lower() == str(system_type_filter).lower())
        sub = chunk.loc[mask, ["Official Symbol Interactor A", "Official Symbol Interactor B"]]
        for a, b in zip(sub["Official Symbol Interactor A"], sub["Official Symbol Interactor B"]):
            sa = _normalize_gene_symbol(a)
            sb = _normalize_gene_symbol(b)
            if sa and sb and sa != sb:
                pairs.add((sa, sb) if sa < sb else (sb, sa))

    df = pd.DataFrame(sorted(pairs), columns=["gene_a", "gene_b"])
    out_path = os.path.join(out_dir, "biogrid_human_ppi.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[PKB] 已保存 BioGRID 人类物理互作: {out_path} (去重边={len(df)}, 覆盖基因={len(set(df['gene_a']) | set(df['gene_b']))})")
    return df


def build_pathway_crosstalk_network(
    membership_df: pd.DataFrame,
    out_dir: str,
    min_shared_genes: int = 3,
    min_jaccard: float = 0.02,
    top_hubs_per_edge: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    通过共享基因自动串联全量通路，构建“通路–通路交错网（Pathway Cross-Talk Network）”并输出全量通路目录。
    """
    os.makedirs(out_dir, exist_ok=True)
    pw_to_genes = membership_df.groupby("pathway_id")["gene_symbol"].apply(set).to_dict()
    pw_meta = membership_df[["pathway_id", "pathway_name", "source_db"]].drop_duplicates().set_index("pathway_id")

    gene_to_pws = defaultdict(list)
    for pw_id, gset in pw_to_genes.items():
        for g in gset:
            gene_to_pws[g].append(pw_id)

    # 通过共享基因倒排索引高效统计通路对交集（避免 O(P^2) 全扫描）
    pair_shared = defaultdict(list)
    for g, pws in gene_to_pws.items():
        if len(pws) < 2:
            continue
        pws_sorted = sorted(pws)
        for i in range(len(pws_sorted)):
            pi = pws_sorted[i]
            for j in range(i + 1, len(pws_sorted)):
                pair_shared[(pi, pws_sorted[j])].append(g)

    edges = []
    pw_degree = defaultdict(int)
    for (p1, p2), shared in pair_shared.items():
        n_shared = len(shared)
        if n_shared < int(min_shared_genes):
            continue
        u = len(pw_to_genes[p1]) + len(pw_to_genes[p2]) - n_shared
        jacc = float(n_shared / max(u, 1))
        if jacc < float(min_jaccard):
            continue
        edges.append((
            p1,
            p2,
            n_shared,
            round(jacc, 4),
            ";".join(sorted(shared)[: int(top_hubs_per_edge)]),
        ))
        pw_degree[p1] += 1
        pw_degree[p2] += 1

    edge_df = pd.DataFrame(
        edges,
        columns=["pathway_a", "pathway_b", "shared_gene_count", "jaccard", "shared_hub_genes"],
    ).sort_values(["shared_gene_count", "jaccard"], ascending=False).reset_index(drop=True)

    catalog_rows = []
    for pw_id, gset in pw_to_genes.items():
        meta = pw_meta.loc[pw_id]
        catalog_rows.append((
            pw_id,
            meta["pathway_name"],
            meta["source_db"],
            len(gset),
            int(pw_degree.get(pw_id, 0)),
        ))
    catalog_df = pd.DataFrame(
        catalog_rows,
        columns=["pathway_id", "pathway_name", "source_db", "gene_count", "crosstalk_degree"],
    ).sort_values(["crosstalk_degree", "gene_count"], ascending=False).reset_index(drop=True)

    edge_path = os.path.join(out_dir, "pathway_crosstalk_edges.parquet")
    cat_path = os.path.join(out_dir, "pathway_catalog.csv")
    edge_df.to_parquet(edge_path, index=False)
    catalog_df.to_csv(cat_path, index=False, encoding="utf-8-sig")
    print(f"[PKB] 已构建通路交错网: {edge_path} (通路间交错边={len(edge_df)}), 全量通路目录: {cat_path} (通路总数={len(catalog_df)})")
    return edge_df, catalog_df


def build_full_prior_atlas(
    out_dir: str,
    biogrid_path: str,
    library_names: str = "KEGG_2026;Reactome_Pathways_2024;WikiPathways_2024_Human;MSigDB_Hallmark_2020;TRRUST_Transcription_Factors_2019",
    min_genes_per_pathway: int = 5,
    max_genes_per_pathway: int = 1000,
    min_shared_genes: int = 3,
    min_jaccard: float = 0.02,
    check_h5ad_path: str = "",
) -> dict:
    """
    一键构建并质检全量人类“通路-基因-有向调控-交错网”先验图谱底座。
    """
    membership_df = fetch_gseapy_pathway_libraries(
        library_names=library_names,
        out_dir=out_dir,
        min_genes_per_pathway=min_genes_per_pathway,
        max_genes_per_pathway=max_genes_per_pathway,
    )
    directed_df = fetch_omnipath_directed_edges(out_dir=out_dir)
    biogrid_df = parse_biogrid_human_ppi(biogrid_path=biogrid_path, out_dir=out_dir)
    crosstalk_df, catalog_df = build_pathway_crosstalk_network(
        membership_df=membership_df,
        out_dir=out_dir,
        min_shared_genes=min_shared_genes,
        min_jaccard=min_jaccard,
    )

    pw_genes = set(membership_df["gene_symbol"].unique())
    dir_genes = set(directed_df["src"].unique()) | set(directed_df["dst"].unique())
    bio_genes = set(biogrid_df["gene_a"].unique()) | set(biogrid_df["gene_b"].unique())
    all_prior_genes = pw_genes | dir_genes | bio_genes

    summary = {
        "total_pathways": int(len(catalog_df)),
        "pathways_by_db": catalog_df["source_db"].value_counts().to_dict(),
        "membership_edges": int(len(membership_df)),
        "pathway_unique_genes": int(len(pw_genes)),
        "directed_signaling_edges": int(len(directed_df)),
        "directed_activation_edges": int((directed_df["sign"] == 1).sum()),
        "directed_inhibition_edges": int((directed_df["sign"] == -1).sum()),
        "biogrid_physical_edges": int(len(biogrid_df)),
        "pathway_crosstalk_edges": int(len(crosstalk_df)),
        "all_prior_unique_genes": int(len(all_prior_genes)),
    }

    if check_h5ad_path and os.path.exists(check_h5ad_path):
        import anndata as ad
        ad_obj = ad.read_h5ad(check_h5ad_path, backed="r")
        sc_genes = {_normalize_gene_symbol(g) for g in ad_obj.var_names} - {""}
        ad_obj.file.close()
        summary["check_h5ad_genes"] = int(len(sc_genes))
        summary["h5ad_covered_by_pathway"] = int(len(sc_genes & pw_genes))
        summary["h5ad_covered_by_directed"] = int(len(sc_genes & dir_genes))
        summary["h5ad_covered_by_any_prior"] = int(len(sc_genes & all_prior_genes))

    sum_path = os.path.join(out_dir, "prior_atlas_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[PKB] 先验底座总览已保存: {sum_path}")
    return summary

# 中文注释：本模块把全量人类信号/代谢/转录通路与 OmniPath 有向调控、BioGRID 物理互作统一整理为
# “基因-通路二部图 + 通路交错网”底座，严禁只挑少数通路，确保后续虚拟细胞底座拥有完整的生物拓扑先验。
