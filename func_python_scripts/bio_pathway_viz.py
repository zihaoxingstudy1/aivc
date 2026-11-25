# -*- coding: utf-8 -*-
"""
下游生信机制解释与发表级可视化引擎 (GO/KEGG Pathway Enrichment & Regulatory Network Visualization)
1. 对模型预测的基因扰动/药物响应 Delta 向量提取显著上调/下调基因集，执行 GO Biological Process & KEGG 通路富集 (gseapy.enrichr)；
2. 生成科研级 GO/KEGG 通路富集气泡图 (Bubble Plot)；
3. 构建并绘制“扰动靶点 (Driver/Passenger) -> 核心响应基因 (Up/Down) -> 富集 KEGG/GO 通路”三层调控网络图 (NetworkX)。
"""

import os
import numpy as np
import pandas as pd
import gseapy as gp
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def run_go_kegg_enrichment(
    pred_delta: np.ndarray,
    gene_names: list,
    pert_label: str,
    out_csv: str,
    out_bubble_png: str,
    top_n_genes: int = 80,
    gene_sets: tuple = ("KEGG_2021_Human", "GO_Biological_Process_2023"),
) -> pd.DataFrame:
    """对预测的转录组变化 Delta 执行上调/下调基因的 GO & KEGG 富集分析并绘气泡图。"""
    order = np.argsort(pred_delta)
    down_genes = [str(gene_names[i]) for i in order[:top_n_genes]]
    up_genes = [str(gene_names[i]) for i in order[-top_n_genes:][::-1]]

    all_res = []
    for direction, glist in [("Up-regulated", up_genes), ("Down-regulated", down_genes)]:
        try:
            enr = gp.enrichr(
                gene_list=glist,
                gene_sets=list(gene_sets),
                organism="human",
                outdir=None,
                no_plot=True,
            )
            df = enr.results.copy()
            if len(df) > 0:
                df["Direction"] = direction
                df["Perturbation"] = pert_label
                all_res.append(df)
        except Exception as e:
            print(f"[Enrichr Warning] {pert_label} ({direction}): {e}")

    if not all_res:
        return pd.DataFrame()

    res_df = pd.concat(all_res, ignore_index=True).sort_values("Adjusted P-value")
    res_df.to_csv(out_csv, index=False, encoding="utf-8")

    # 绘制 Top 16 显著富集通路气泡图 (按 KEGG / GO_BP 分组展示)
    plot_df = res_df.head(16).copy()
    plot_df["neg_log10_fdr"] = -np.log10(np.clip(plot_df["Adjusted P-value"].values, 1e-30, 1.0))
    plot_df["GeneCount"] = plot_df["Overlap"].astype(str).map(lambda x: int(x.split("/")[0]) if "/" in x else 1)
    plot_df["ShortTerm"] = plot_df["Term"].astype(str).map(lambda x: x.split(" (GO:")[0][:48])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
    palette = {"Up-regulated": "#e74c3c", "Down-regulated": "#2980b9"}
    sns.scatterplot(
        data=plot_df,
        x="neg_log10_fdr",
        y="ShortTerm",
        hue="Direction",
        size="GeneCount",
        sizes=(80, 420),
        palette=palette,
        alpha=0.85,
        ax=ax,
    )
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1.0, label="FDR = 0.05")
    ax.set_title(f"Predicted Transcriptomic GO & KEGG Enrichment: {pert_label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("-log10 (Adjusted P-value / FDR)", fontsize=10)
    ax.set_ylabel("")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_bubble_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Enrichr] {pert_label} 富集完成 ({len(res_df)} 条通路)，已保存至 {out_csv} & {out_bubble_png}")
    return res_df


def plot_target_pathway_network(
    target_names: list,
    target_shares: list,
    pred_delta: np.ndarray,
    gene_names: list,
    enrich_df: pd.DataFrame,
    pert_label: str,
    out_net_png: str,
    top_pathways: int = 6,
    max_genes_per_pathway: int = 5,
):
    """绘制 靶点(Driver/Passenger) -> 核心响应基因(Up/Down) -> 显著通路 三层调控网络图。"""
    if enrich_df is None or len(enrich_df) == 0:
        return

    delta_map = {str(g): float(pred_delta[i]) for i, g in enumerate(gene_names)}
    G = nx.DiGraph()

    # 第 1 层: 扰动靶点节点
    for tg, sh in zip(target_names, target_shares):
        G.add_node(tg, layer=0, ntype="Target", label=f"{tg}\n({sh*100:.1f}%)")

    # 第 3 层: Top 显著通路及其第 2 层核心基因
    sub_pw = enrich_df.head(top_pathways)
    added_genes = set()
    for _, row in sub_pw.iterrows():
        pw_name = str(row["Term"]).split(" (GO:")[0][:34]
        pw_dir = str(row.get("Direction", "Up-regulated"))
        G.add_node(pw_name, layer=2, ntype="Pathway", direction=pw_dir, label=pw_name)

        genes_in_pw = [g.strip() for g in str(row.get("Genes", "")).split(";") if g.strip() in delta_map]
        # 按绝对变化幅度排序取 Top 基因
        genes_in_pw = sorted(genes_in_pw, key=lambda g: abs(delta_map[g]), reverse=True)[:max_genes_per_pathway]
        for g in genes_in_pw:
            dv = delta_map[g]
            if g not in added_genes:
                G.add_node(g, layer=1, ntype="Gene", delta=dv, label=f"{g}\n({dv:+.2f})")
                added_genes.add(g)
                # 连接主导靶点到核心基因
                for tg, sh in zip(target_names, target_shares):
                    if sh >= 0.25 or len(target_names) == 1:
                        G.add_edge(tg, g, weight=sh)
            G.add_edge(g, pw_name, weight=abs(dv))

    # 分层布局
    pos = {}
    layers = {0: [], 1: [], 2: []}
    for n, d in G.nodes(data=True):
        layers[d["layer"]].append(n)

    for l_idx, x_val in [(0, 0.0), (1, 1.0), (2, 2.0)]:
        nodes_l = layers[l_idx]
        ys = np.linspace(1.0, -1.0, len(nodes_l)) if len(nodes_l) > 1 else [0.0]
        for n, y_val in zip(nodes_l, ys):
            pos[n] = (x_val, float(y_val))

    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=180)
    node_colors, node_sizes = [], []
    for n, d in G.nodes(data=True):
        if d["ntype"] == "Target":
            node_colors.append("#8e44ad")
            node_sizes.append(1800)
        elif d["ntype"] == "Gene":
            node_colors.append("#e74c3c" if d["delta"] > 0 else "#2980b9")
            node_sizes.append(950)
        else:
            node_colors.append("#e67e22" if d["direction"] == "Up-regulated" else "#16a085")
            node_sizes.append(1500)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bdc3c7", alpha=0.65, arrows=True, arrowsize=12, width=1.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.92, edgecolors="white", linewidths=1.5)
    labels = {n: d["label"] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7.5, font_weight="bold", ax=ax)

    ax.set_title(f"Target → Gene → Pathway Regulatory Network: {pert_label}\n(Purple=Target, Red/Blue=Up/Down Gene, Orange/Teal=Up/Down Pathway)", fontsize=11.5, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_net_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Network] 调控网络图已保存至 {out_net_png}")


def plot_drug_target_pathway_sankey(
    source_name: str,
    targets: list,
    target_shares: list,
    pred_delta: np.ndarray,
    gene_names: list,
    enrich_df: pd.DataFrame,
    out_sankey_png: str,
    top_pathways: int = 5,
):
    """
    绘制发表级 4 层桑基/冲积流向图 (Alluvial / Sankey Flow):
    [药物/组合源] -> [核心靶点 (Driver/Passenger)] -> [Top 上/下调基因] -> [富集 KEGG/GO 通路]
    """
    from matplotlib.path import Path
    import matplotlib.patches as patches

    if enrich_df is None or len(enrich_df) == 0:
        return

    delta_map = {str(g): float(pred_delta[i]) for i, g in enumerate(gene_names)}
    sub_pw = enrich_df.head(top_pathways).copy()

    # 收集 Top 基因 (每个通路最多取 2 个最强基因，去重最多 8 个)
    pw_genes = []
    seen_g = set()
    for _, r in sub_pw.iterrows():
        pw_short = str(r["Term"]).split(" (GO:")[0][:30]
        gs = [g.strip() for g in str(r.get("Genes", "")).split(";") if g.strip() in delta_map]
        gs = sorted(gs, key=lambda g: abs(delta_map[g]), reverse=True)[:2]
        for g in gs:
            pw_genes.append((g, pw_short, abs(delta_map[g]), str(r.get("Direction", "Up-regulated"))))
            seen_g.add(g)

    genes_sorted = sorted(seen_g, key=lambda g: delta_map[g], reverse=True)[:8]
    pw_list = [str(r["Term"]).split(" (GO:")[0][:30] for _, r in sub_pw.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=180)
    ax.set_xlim(-0.2, 3.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    def draw_ribbon(x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.32):
        xm = (x0 + x1) / 2.0
        verts = [
            (x0, y0_bot), (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
            (x1, y1_top), (xm, y1_top), (xm, y0_top), (x0, y0_top), (x0, y0_bot),
        ]
        codes = [
            Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY,
        ]
        ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))

    def layout_boxes(items, weights, x, width=0.14, gap=0.025):
        tot = sum(weights) + 1e-8
        usable = 1.0 - gap * max(len(items) - 1, 0)
        y_cur = 1.0
        boxes = {}
        for it, w in zip(items, weights):
            h = max(0.03, usable * (w / tot))
            boxes[it] = (x, y_cur - h, y_cur)
            y_cur -= (h + gap)
        return boxes

    b0 = layout_boxes([source_name], [1.0], x=0.05)
    b1 = layout_boxes(targets, target_shares, x=0.95)
    b2 = layout_boxes(genes_sorted, [abs(delta_map[g]) for g in genes_sorted], x=1.90)
    b3 = layout_boxes(pw_list, [1.0] * len(pw_list), x=2.85)

    # 绘制方块与文字
    for it, (x, yb, yt) in b0.items():
        ax.add_patch(patches.Rectangle((x, yb), 0.14, yt - yb, facecolor="#34495e", edgecolor="white"))
        ax.text(x + 0.07, (yb + yt) / 2, it, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    for it, sh in zip(targets, target_shares):
        x, yb, yt = b1[it]
        ax.add_patch(patches.Rectangle((x, yb), 0.14, yt - yb, facecolor="#8e44ad", edgecolor="white"))
        ax.text(x + 0.07, (yb + yt) / 2, f"{it}\n({sh*100:.1f}%)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    for g in genes_sorted:
        x, yb, yt = b2[g]
        c = "#e74c3c" if delta_map[g] > 0 else "#2980b9"
        ax.add_patch(patches.Rectangle((x, yb), 0.14, yt - yb, facecolor=c, edgecolor="white"))
        ax.text(x + 0.07, (yb + yt) / 2, f"{g} ({delta_map[g]:+.2f})", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    for _, r in sub_pw.iterrows():
        pw = str(r["Term"]).split(" (GO:")[0][:30]
        x, yb, yt = b3[pw]
        c = "#e67e22" if str(r.get("Direction", "")) == "Up-regulated" else "#16a085"
        ax.add_patch(patches.Rectangle((x, yb), 0.14, yt - yb, facecolor=c, edgecolor="white"))
        ax.text(x + 0.16, (yb + yt) / 2, pw, ha="left", va="center", color="#2c3e50", fontsize=8.5, fontweight="bold")

    # 绘制流向带: L0->L1, L1->L2, L2->L3
    sx, syb, syt = b0[source_name]
    y_ptr = syt
    for tg, sh in zip(targets, target_shares):
        tx, tyb, tyt = b1[tg]
        span = (syt - syb) * sh
        draw_ribbon(sx + 0.14, y_ptr - span, y_ptr, tx, tyb, tyt, "#8e44ad", alpha=0.35)
        y_ptr -= span

    for tg, sh in zip(targets, target_shares):
        tx, tyb, tyt = b1[tg]
        for g in genes_sorted:
            gx, gyb, gyt = b2[g]
            c = "#e74c3c" if delta_map[g] > 0 else "#2980b9"
            draw_ribbon(tx + 0.14, tyb, tyt, gx, gyb, gyt, c, alpha=0.14 * sh + 0.08)

    for g, pw, w, dstr in pw_genes:
        if g in b2 and pw in b3:
            gx, gyb, gyt = b2[g]
            px, pyb, pyt = b3[pw]
            c = "#e67e22" if dstr == "Up-regulated" else "#16a085"
            draw_ribbon(gx + 0.14, gyb, gyt, px, pyb, pyt, c, alpha=0.32)

    for col_x, title in [(0.12, "Perturbation / Drug"), (1.02, "Target (Driver/Pass.)"), (1.97, "Key DEGs (Delta)"), (2.92, "GO / KEGG Pathways")]:
        ax.text(col_x, 1.03, title, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2c3e50")

    ax.set_title(f"Mechanism Flow Sankey Diagram: {source_name} → Targets → Genes → Pathways", fontsize=12.5, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(out_sankey_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sankey] 机制流向桑基图已保存至 {out_sankey_png}")


def plot_mechanistic_signaling_map(
    drug_name: str,
    direct_targets: list,
    pathway_nodes: list,
    edges: list,
    downstream_outcomes: list,
    out_png: str,
):
    """
    Plot an interpretable 4-stage Mechanistic Signaling Pathway Map:
      Stage 1: Drug -> Stage 2: Direct Targets -> Stage 3: Pathway Hub Proteins & Phosphosites
      (colored Red for predicted UP ↑, Blue for predicted DOWN ↓, annotated with pathway tier)
      -> Stage 4: Downstream Biological / Clinical Pathways.

    Parameters
    ----------
    drug_name : str
        Name of the drug or combination.
    direct_targets : list of tuple
        [(target_name, weight_or_role), ...]
    pathway_nodes : list of dict
        [{"name": str, "tier": "Upstream"|"Mid_Kinase"|"Downstream", "delta": float, "is_phospho": bool}, ...]
    edges : list of tuple
        [(src_name, dst_name, edge_type), ...] where edge_type in ("activate", "inhibit", "bind")
    downstream_outcomes : list of tuple
        [(pathway_or_outcome_name, direction_str), ...]
    out_png : str
        Output PNG file path.
    """
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=200)
    ax.axis("off")

    G = nx.DiGraph()
    pos = {}
    node_meta = {}

    # Column 0: Drug
    G.add_node(drug_name)
    pos[drug_name] = (0.0, 0.5)
    node_meta[drug_name] = {"kind": "drug", "delta": 0.0, "label": drug_name}

    # Column 1: Direct Targets
    n_t = max(len(direct_targets), 1)
    for i, (tname, role) in enumerate(direct_targets):
        y = 0.5 if n_t == 1 else (0.82 - i * (0.64 / (n_t - 1)))
        G.add_node(tname)
        pos[tname] = (1.6, y)
        node_meta[tname] = {"kind": "target", "delta": 0.0, "label": f"{tname}\n({role})"}
        G.add_edge(drug_name, tname, etype="bind")

    # Columns 2, 3, 4: Pathway Tiers (Upstream -> Mid_Kinase -> Downstream)
    tier_x = {"Upstream": 3.4, "Mid_Kinase": 5.3, "Downstream": 7.2}
    for tier, x_val in tier_x.items():
        tnodes = [nd for nd in pathway_nodes if nd.get("tier") == tier]
        nt = max(len(tnodes), 1)
        for i, nd in enumerate(tnodes):
            y = 0.5 if nt == 1 else (0.88 - i * (0.76 / (nt - 1)))
            nm = nd["name"]
            dval = float(nd.get("delta", 0.0))
            arrow = "↑" if dval >= 0 else "↓"
            tag = "[P]" if nd.get("is_phospho", False) else "[Prot]"
            G.add_node(nm)
            pos[nm] = (x_val, y)
            node_meta[nm] = {
                "kind": "hub",
                "delta": dval,
                "label": f"{nm} {tag}\n{arrow} {dval:+.2f}",
            }

    # Column 5: Downstream Pathways / Outcomes
    n_o = max(len(downstream_outcomes), 1)
    for i, (oname, odir) in enumerate(downstream_outcomes):
        y = 0.5 if n_o == 1 else (0.85 - i * (0.70 / (n_o - 1)))
        G.add_node(oname)
        pos[oname] = (9.2, y)
        node_meta[oname] = {"kind": "outcome", "delta": 1.0 if "Up" in odir or "Act" in odir else -1.0, "label": f"{oname}\n({odir})"}

    for s, d, et in edges:
        if s in G.nodes and d in G.nodes:
            G.add_edge(s, d, etype=et)

    # Draw edges by type
    for u, v, edata in G.edges(data=True):
        et = edata.get("etype", "activate")
        col = "#7f8c8d" if et == "bind" else ("#c0392b" if et == "inhibit" else "#2980b9")
        style = "dashed" if et == "inhibit" else "solid"
        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, linestyle=style, shrinkA=22, shrinkB=22, alpha=0.75),
            zorder=1,
        )

    # Draw nodes
    for n, (x, y) in pos.items():
        meta = node_meta[n]
        kind = meta["kind"]
        dval = meta["delta"]
        if not np.isfinite(dval):
            dval = 0.0
        if kind == "drug":
            fc, ec, sz = "#8e44ad", "#4a235a", 1800
        elif kind == "target":
            fc, ec, sz = "#f39c12", "#7e5109", 1400
        elif kind == "outcome":
            fc = "#f5b7b1" if dval >= 0 else "#aed6f1"
            ec, sz = "#2c3e50", 1400
        else:
            fc = "#e74c3c" if dval >= 0 else "#3498db"
            ec = "#78281f" if dval >= 0 else "#1b4f72"
            sz = 950 + min(abs(dval) * 1200, 1200)

        ax.scatter([x], [y], s=sz, c=fc, edgecolors=ec, linewidths=1.8, zorder=3, alpha=0.92)
        txt_col = "white" if kind in ("drug", "hub") else "#1c2833"
        ax.text(x, y, meta["label"], ha="center", va="center", fontsize=8.0, fontweight="bold", color=txt_col, zorder=4)

    # Column headers
    headers = [
        (0.0, "1. Drug"),
        (1.6, "2. Direct Target"),
        (3.4, "3a. Upstream RTK/Node"),
        (5.3, "3b. Kinase / Phospho Hub"),
        (7.2, "3c. Effector Protein"),
        (9.2, "4. Pathway Impact"),
    ]
    for hx, htxt in headers:
        ax.text(hx, 0.98, htxt, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2c3e50")

    ax.set_xlim(-0.8, 10.3)
    ax.set_ylim(-0.02, 1.06)
    ax.set_title(
        f"Pathway-GNN Mechanistic Signaling Map: {drug_name}\n"
        "(Red ↑ = Predicted Protein/Phospho Increase | Blue ↓ = Predicted Decrease | Dashed = Inhibition)",
        fontsize=12, fontweight="bold", pad=12,
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[PathwayMap] 机制信号通路图已保存至 {out_png}")
