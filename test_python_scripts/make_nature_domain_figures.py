# -*- coding: utf-8 -*-
"""
make_nature_domain_figures.py
Generate 4 domain-specific publication-grade Nature figures from 100% real project data:
  1. nature_domain_fig1_scrna_umap_dotplot.png/.pdf   (Single-cell UMAP + perturbation vector field + Marker DotPlot)
  2. nature_domain_fig2_causal_pathway_net.png/.pdf   (OmniPath directed causal signaling network + kinase phospho recovery)
  3. nature_domain_fig3_ode_phase_dynamics.png/.pdf   (18-day reprogramming 2D phase-plane velocity + temporal cascade)
  4. nature_domain_fig4_human_organ_atlas.png/.pdf    (54-organ GTEx physiological system safety atlas & dotplot)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.patches import FancyArrowPatch, Circle
from sklearn.decomposition import PCA

OUT_DIR = r"D:\data\AIVC\test_results"
DATA_DIR = r"D:\data\AIVC\data"
SRC_DIR = r"D:\data\AIVC\source_data"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def save_both(fig, stem):
    png = os.path.join(OUT_DIR, f"{stem}.png")
    pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.png & .pdf")


# -------------------------------------------------------------------------
# 1. Single-Cell Transcriptome UMAP + Perturbation Vector Field + DotPlot
# -------------------------------------------------------------------------
def make_scrna_figure():
    print("[1/4] Building Single-Cell UMAP & Marker DotPlot from Norman2019...")
    ad = sc.read_h5ad(os.path.join(DATA_DIR, "Norman2019_hvg2000.h5ad"))
    # Clean perturbation labels
    p_raw = ad.obs["perturbation"].astype(str).tolist()
    clean_p = []
    for x in p_raw:
        if "negctrl" in x.lower() or x.strip().lower() == "control":
            clean_p.append("Control")
        else:
            parts = [p for p in x.replace(";", "_").split("_") if p and "NegCtrl" not in p]
            uniq = []
            for p in parts:
                if p not in uniq:
                    uniq.append(p)
            clean_p.append("+".join(uniq[:2]))
    ad.obs["prog"] = clean_p

    # Pick major biologically distinct programs + Control
    top_progs = ["Control", "KLF1", "CEBPA", "CEBPE", "SPI1", "GATA1", "RUNX1T1", "CEBPE+RUNX1T1", "CEBPA+KLF1"]
    avail = [p for p in top_progs if (ad.obs["prog"] == p).sum() >= 60]
    if len(avail) < 6:
        vc = ad.obs["prog"].value_counts()
        avail = ["Control"] + [k for k in vc.index if k != "Control"][:7]

    rng = np.random.RandomState(42)
    keep_idx = []
    for p in avail:
        idx = np.where(ad.obs["prog"].values == p)[0]
        take = rng.choice(idx, size=min(len(idx), 650), replace=False)
        keep_idx.extend(take.tolist())
    sub = ad[keep_idx].copy()
    sc.pp.pca(sub, n_comps=30)
    sc.pp.neighbors(sub, n_neighbors=20, n_pcs=30)
    sc.tl.umap(sub, random_state=42)

    fig = plt.figure(figsize=(15.5, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.25], wspace=0.28)
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])

    pal = {
        "Control": "#BDC3C7", "KLF1": "#2B5B84", "CEBPA": "#C0392B",
        "CEBPE": "#E67E22", "SPI1": "#16A085", "GATA1": "#8E44AD",
        "RUNX1T1": "#2980B9", "CEBPE+RUNX1T1": "#D35400", "CEBPA+KLF1": "#1ABC9C"
    }
    umap_xy = sub.obsm["X_umap"]

    # Panel A: Single-cell UMAP colored by perturbation program
    for p in avail:
        m = sub.obs["prog"].values == p
        alpha = 0.40 if p == "Control" else 0.78
        sz = 7 if p == "Control" else 10
        ax_u.scatter(umap_xy[m, 0], umap_xy[m, 1], s=sz, c=pal.get(p, "#7F8C8D"),
                     alpha=alpha, edgecolors="none", label=p, rasterized=True)
    ax_u.set_xlabel("UMAP 1", fontsize=9.2, fontweight="bold")
    ax_u.set_ylabel("UMAP 2", fontsize=9.2, fontweight="bold")
    ax_u.set_title("a  Single-Cell Perturbation Manifold (Norman2019)", loc="left", fontsize=10.2, fontweight="bold")
    ax_u.spines["top"].set_visible(False)
    ax_u.spines["right"].set_visible(False)
    ax_u.legend(fontsize=7.3, markerscale=1.7, frameon=True, ncol=2, loc="best")

    # Panel B: Perturbation Displacement Vector Field (Control -> Perturbed Centroids)
    ctrl_m = sub.obs["prog"].values == "Control"
    c0 = umap_xy[ctrl_m].mean(axis=0)
    ax_v.scatter(umap_xy[:, 0], umap_xy[:, 1], s=6, c="#E5E8E8", alpha=0.65, edgecolors="none", rasterized=True)
    ax_v.scatter([c0[0]], [c0[1]], s=160, c="#2C3E50", marker="*", zorder=6, label="Unperturbed Control")
    for p in avail:
        if p == "Control":
            continue
        m = sub.obs["prog"].values == p
        cp = umap_xy[m].mean(axis=0)
        col = pal.get(p, "#2B5B84")
        ax_v.quiver(c0[0], c0[1], cp[0] - c0[0], cp[1] - c0[1],
                    angles="xy", scale_units="xy", scale=1.0,
                    color=col, width=0.007, headwidth=4.0, headlength=5.0, zorder=5)
        ax_v.scatter([cp[0]], [cp[1]], s=85, c=col, edgecolors="#111111", lw=0.9, zorder=6)
        ax_v.text(cp[0] + 0.18, cp[1] + 0.12, p, fontsize=7.6, fontweight="bold", color=col, zorder=7)
    ax_v.set_xlabel("UMAP 1", fontsize=9.2, fontweight="bold")
    ax_v.set_ylabel("UMAP 2", fontsize=9.2, fontweight="bold")
    ax_v.set_title("b  Perturbation Fate Vector Field (Δx Trajectory)", loc="left", fontsize=10.2, fontweight="bold")
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)

    # Panel C: Marker Gene DotPlot (top variable marker genes across programs)
    X_dense = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    var_names = [str(v) for v in sub.var_names]
    ctrl_mean = X_dense[ctrl_m].mean(axis=0)
    # Pick top 12 genes with highest across-program variance
    prog_means = np.vstack([X_dense[sub.obs["prog"].values == p].mean(axis=0) for p in avail])
    top_g_idx = np.argsort(-np.std(prog_means, axis=0))[:12]
    sel_genes = [var_names[i] for i in top_g_idx]

    mean_mat = np.zeros((len(avail), len(sel_genes)))
    frac_mat = np.zeros((len(avail), len(sel_genes)))
    for r_i, p in enumerate(avail):
        m = sub.obs["prog"].values == p
        Xp = X_dense[m][:, top_g_idx]
        mean_mat[r_i] = Xp.mean(axis=0) - ctrl_mean[top_g_idx]
        frac_mat[r_i] = (Xp > 0).mean(axis=0) * 100.0

    # Z-score columns for crisp contrast
    col_std = np.std(mean_mat, axis=0) + 1e-6
    z_mat = np.clip((mean_mat - mean_mat.mean(axis=0)) / col_std, -2.2, 2.2)

    xs, ys, cs, ss = [], [], [], []
    for r_i in range(len(avail)):
        for c_i in range(len(sel_genes)):
            xs.append(c_i)
            ys.append(r_i)
            cs.append(z_mat[r_i, c_i])
            ss.append(15 + 1.45 * frac_mat[r_i, c_i])
    sc_dot = ax_d.scatter(xs, ys, c=cs, s=ss, cmap="RdBu_r", vmin=-2.0, vmax=2.0,
                          edgecolors="#222222", lw=0.5)
    ax_d.set_xticks(range(len(sel_genes)))
    ax_d.set_xticklabels(sel_genes, rotation=40, ha="right", fontsize=7.8, fontweight="bold")
    ax_d.set_yticks(range(len(avail)))
    ax_d.set_yticklabels(avail, fontsize=8.0, fontweight="bold")
    ax_d.set_title("c  Program Marker DotPlot (Size=%Expr, Color=Z-Δ)", loc="left", fontsize=10.2, fontweight="bold")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.grid(True, linestyle=":", alpha=0.35)
    cbar = fig.colorbar(sc_dot, ax=ax_d, shrink=0.72, pad=0.03)
    cbar.set_label("Relative ΔExpr (Z)", fontsize=8.0)

    fig.suptitle("Single-Cell Perturbation Landscape, Vector Displacement & Program Markers",
                 fontsize=12.0, fontweight="bold", y=1.03)
    save_both(fig, "nature_domain_fig1_scrna_umap_dotplot")


# -------------------------------------------------------------------------
# 2. OmniPath Causal Signaling Network + Kinase Phospho Recovery
# -------------------------------------------------------------------------
def make_pathway_network_figure():
    print("[2/4] Building OmniPath Causal Pathway Network...")
    edges = pd.read_parquet(os.path.join(DATA_DIR, "priors", "directed_signaling_edges.parquet"))
    scol, tcol, sgcol = edges.columns[0], edges.columns[1], edges.columns[2]
    edges[scol] = edges[scol].astype(str).str.upper()
    edges[tcol] = edges[tcol].astype(str).str.upper()

    drivers = ["EGFR", "ERBB2", "BRAF", "MAP2K1", "AKT1"]
    effectors = ["MAP2K2", "ELK1", "FOXM1", "BAD", "CASP9", "AGO2", "ANXA1", "CDK1", "BECN1", "GSK3B"]
    nodes = drivers + effectors
    sub_e = edges[edges[scol].isin(nodes) & edges[tcol].isin(nodes)].drop_duplicates([scol, tcol])

    # Manual clean layered biological coordinates (Top=Receptors, Mid=Kinases, Bottom=Effectors)
    pos = {
        "EGFR": (-1.5, 2.2), "ERBB2": (1.5, 2.2),
        "BRAF": (-2.0, 1.0), "AKT1": (2.0, 1.0),
        "MAP2K1": (-0.7, 0.0), "MAP2K2": (0.7, 0.0),
        "ELK1": (-3.0, -1.2), "FOXM1": (-2.0, -1.2), "GSK3B": (-1.0, -1.2),
        "BAD": (0.0, -1.2), "CASP9": (1.0, -1.2), "CDK1": (2.0, -1.2),
        "AGO2": (-2.6, 1.15), "ANXA1": (0.0, 1.15), "BECN1": (3.0, -1.2),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]

    # Draw directed causal edges (blue=activation, red=inhibition)
    for _, r in sub_e.iterrows():
        u, v, sgn = r[scol], r[tcol], float(r[sgcol])
        if u not in pos or v not in pos or u == v:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        col = "#2B5B84" if sgn > 0 else "#C0392B"
        ls = "-" if sgn > 0 else "--"
        rad = 0.14 if abs(x1 - x2) > 1.5 else 0.06
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              connectionstyle=f"arc3,rad={rad}",
                              arrowstyle="-|>,head_length=5,head_width=3" if sgn > 0 else "-[,widthB=3.5,lengthB=1.5",
                              color=col, lw=1.6, linestyle=ls, alpha=0.82, shrinkA=16, shrinkB=16, zorder=2)
        ax.add_patch(arr)

    # Draw nodes by tier
    for nd, (x, y) in pos.items():
        if nd in ("EGFR", "ERBB2"):
            nc, sz = "#C0392B", 980
        elif nd in ("BRAF", "MAP2K1", "AKT1", "MAP2K2"):
            nc, sz = "#2B5B84", 880
        else:
            nc, sz = "#16A085", 740
        ax.scatter([x], [y], s=sz, c=nc, edgecolors="#111111", lw=1.3, zorder=4)
        ax.text(x, y, nd, ha="center", va="center", fontsize=6.6, fontweight="bold", color="white", zorder=5)

    # Legend proxies
    ax.plot([], [], color="#2B5B84", lw=2.0, label="Causal Activation (+)")
    ax.plot([], [], color="#C0392B", lw=2.0, ls="--", label="Causal Inhibition (-)")
    ax.scatter([], [], s=110, c="#C0392B", edgecolors="#111", label="RTK Driver")
    ax.scatter([], [], s=110, c="#2B5B84", edgecolors="#111", label="Core Kinase Hub")
    ax.scatter([], [], s=110, c="#16A085", edgecolors="#111", label="Downstream Effector")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-1.75, 2.75)
    ax.axis("off")
    ax.set_title("a  OmniPath Directed Causal Signaling Topology (Real Edges)", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.0, frameon=True)

    # Panel B: Real Exp35 Masked-Kinase Phosphorylation Recovery Heatmap
    exp35 = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp35_masked_kinase_and_kalman_results.csv"))
    sub35 = exp35[(exp35["part"] == "A_MaskedKinase") & (exp35["subset"] == "Masked_Kinase_Phospho")]
    piv = sub35.pivot_table(index="model", columns="cohort", values="median_r")
    order_m = [m for m in ["Cis_mRNA_MaskedKinase", "Exp28_5Omics_MaskedKinase", "Exp28+OmniPathCausal_MaskedKinase"] if m in piv.index]
    piv = piv.loc[order_m]
    im = axes[1].imshow(piv.values, cmap="YlGnBu", vmin=0.0, vmax=0.25, aspect="auto")
    axes[1].set_xticks(range(len(piv.columns)))
    axes[1].set_xticklabels(piv.columns, fontsize=9.0, fontweight="bold")
    axes[1].set_yticks(range(len(piv.index)))
    lbl_map = {
        "Cis_mRNA_MaskedKinase": "Cis-mRNA Only\n(Masked Kinase)",
        "Exp28_5Omics_MaskedKinase": "5-Omics Base\n(Masked Kinase)",
        "Exp28+OmniPathCausal_MaskedKinase": "AIVC + OmniPath Causal\n(Masked Kinase)",
    }
    axes[1].set_yticklabels([lbl_map.get(x, x) for x in piv.index], fontsize=8.3, fontweight="bold")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            axes[1].text(j, i, f"{val:.4f}", ha="center", va="center",
                         fontsize=10.0, fontweight="bold", color="white" if val > 0.13 else "#111111")
    axes[1].set_title("b  Blind Tumor Masked-Kinase Phospho Recovery (r)", loc="left", fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=axes[1], shrink=0.75, label="Phosphosite Pearson r")

    fig.suptitle("Mechanistic White-Box Causal Pathway Network & Cross-Omics Kinase Recovery",
                 fontsize=12.0, fontweight="bold", y=1.02)
    save_both(fig, "nature_domain_fig2_causal_pathway_net")


# -------------------------------------------------------------------------
# 3. Continuous-Time ODE Phase-Plane Vector Field & Trajectory Ribbon
# -------------------------------------------------------------------------
def make_ode_dynamics_figure():
    print("[3/4] Building Continuous-Time ODE Phase-Plane & Trajectory Ribbon...")
    ck = np.load(os.path.join(DATA_DIR, "schiebinger_18day_pb_hvg2000_cache.npz"), allow_pickle=True)
    days = ck["days"].astype(float)
    deltas = ck["deltas"].astype(float)
    perts = [str(x) for x in ck["perts"]]
    conds = [p.rsplit("_d", 1)[0] for p in perts]

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(deltas)
    var_exp = pca.explained_variance_ratio_ * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=300)
    c_map = {"control": "#2B5B84", "2i (LIF+MEKi+GSK3i)": "#16A085", "Dox": "#C0392B"}

    # Panel A: 2D Phase-Plane Trajectory & Velocity Field
    for cond in sorted(set(conds)):
        idx = [i for i, c in enumerate(conds) if c == cond]
        idx = sorted(idx, key=lambda i: days[i])
        pts = Z[idx]
        ds = days[idx]
        col = c_map.get(cond, "#8E44AD")
        axes[0].plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.9, alpha=0.82, label=f"{cond}")
        # Add velocity arrows every 2nd step for clarity
        for k in range(0, len(pts) - 1, 2):
            dx, dy = pts[k + 1, 0] - pts[k, 0], pts[k + 1, 1] - pts[k, 1]
            axes[0].quiver(pts[k, 0], pts[k, 1], dx, dy, angles="xy", scale_units="xy", scale=1.0,
                           color=col, alpha=0.85, width=0.006, headwidth=3.8)
        axes[0].scatter(pts[0, 0], pts[0, 1], s=85, c=col, marker="o", edgecolors="#111", zorder=5)
        axes[0].scatter(pts[-1, 0], pts[-1, 1], s=110, c=col, marker="^", edgecolors="#111", zorder=5)

    axes[0].set_xlabel(f"State Space PC1 ({var_exp[0]:.1f}% var)", fontsize=9.0, fontweight="bold")
    axes[0].set_ylabel(f"State Space PC2 ({var_exp[1]:.1f}% var)", fontsize=9.0, fontweight="bold")
    axes[0].set_title("a  18-Day Reprogramming Phase-Plane Flow", loc="left", fontsize=10.2, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, ls="--", alpha=0.35)
    axes[0].legend(fontsize=7.8, frameon=True)

    # Panel B: Continuous-Time Program Magnitude Ribbon (Mean +/- SD over top 50 dynamic genes)
    top_dyn = np.argsort(-np.std(deltas, axis=0))[:50]
    mag = np.linalg.norm(deltas[:, top_dyn], axis=1)
    u_days = np.sort(np.unique(days))
    m_vals = [mag[days == d].mean() for d in u_days]
    s_vals = [mag[days == d].std() if (days == d).sum() > 1 else 0.0 for d in u_days]
    m_vals, s_vals = np.array(m_vals), np.array(s_vals)
    axes[1].plot(u_days, m_vals, "-o", color="#2B5B84", lw=2.3, ms=5, label="Dynamic State Trajectory")
    axes[1].fill_between(u_days, m_vals - s_vals, m_vals + s_vals, color="#2B5B84", alpha=0.20, label="±1 SD Across Lineages")
    axes[1].axvspan(12.0, 18.5, color="#16A085", alpha=0.10, label="Blind Forecast Window (D12.5–D18)")
    axes[1].axvline(12.0, color="#16A085", ls="--", lw=1.4)
    axes[1].set_xlabel("Reprogramming Time (Days)", fontsize=9.0, fontweight="bold")
    axes[1].set_ylabel("Top-50 Dynamic Program Norm", fontsize=9.0, fontweight="bold")
    axes[1].set_title("b  Continuous-Time State Ribbon (D0–D18)", loc="left", fontsize=10.2, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(True, ls="--", alpha=0.35)
    axes[1].legend(fontsize=7.5, frameon=True, loc="upper left")

    # Panel C: 0–48h Pathway Cascade Heatmap (real Exp32b data)
    cas = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp32b_temporal_pathway_cascade.csv"))
    piv_c = cas.pivot_table(index="pathway", columns="time_hours", values="mean_abs_activity")
    piv_c = piv_c[sorted(piv_c.columns)]
    piv_c = piv_c.loc[piv_c.mean(axis=1).sort_values(ascending=False).head(10).index]
    im = axes[2].imshow(piv_c.values, cmap="magma", aspect="auto")
    axes[2].set_xticks(range(len(piv_c.columns)))
    axes[2].set_xticklabels([f"{h:g}h" for h in piv_c.columns], fontsize=8.5, fontweight="bold")
    axes[2].set_yticks(range(len(piv_c.index)))
    axes[2].set_yticklabels([str(x)[:28] for x in piv_c.index], fontsize=7.6, fontweight="bold")
    axes[2].set_title("c  0–48h Drug-Response Pathway Cascade", loc="left", fontsize=10.2, fontweight="bold")
    fig.colorbar(im, ax=axes[2], shrink=0.78, label="Mean |ΔExpr|")

    fig.suptitle("Grey-Box Temporal Dynamics: 18-Day Phase-Plane Flow & 0–48h Pathway Cascade",
                 fontsize=12.0, fontweight="bold", y=1.03)
    save_both(fig, "nature_domain_fig3_ode_phase_dynamics")


# -------------------------------------------------------------------------
# 4. Human 54-Organ GTEx Translational Safety Atlas
# -------------------------------------------------------------------------
def make_organ_safety_atlas():
    print("[4/4] Building Human 54-Organ GTEx Translational Safety Atlas...")
    gtex = pd.read_csv(os.path.join(DATA_DIR, "GTEx_v8_gene_median_tpm.gct.gz"), sep="\t", skiprows=2)
    targets = ["GLP1R", "RPE65", "ERBB2", "MAP2K1", "EGFR", "BRAF"]
    sub = gtex[gtex["Description"].isin(targets)].groupby("Description").median(numeric_only=True)
    sub = sub.loc[[g for g in targets if g in sub.index]]
    organs = list(sub.columns)

    def map_sys(o):
        ol = o.lower()
        if "brain" in ol or "nerve" in ol or "pituitary" in ol:
            return "CNS & Nerve"
        if "heart" in ol or "artery" in ol:
            return "Cardiovascular"
        if "liver" in ol or "stomach" in ol or "colon" in ol or "esophagus" in ol or "pancreas" in ol or "small intestine" in ol:
            return "GI & Metabolic"
        if "lung" in ol:
            return "Respiratory"
        if "blood" in ol or "spleen" in ol or "lymph" in ol:
            return "Immune & Blood"
        if "kidney" in ol or "bladder" in ol:
            return "Renal & Urinary"
        if "ovary" in ol or "testis" in ol or "uterus" in ol or "prostate" in ol or "vagina" in ol or "cervix" in ol or "fallopian" in ol or "breast" in ol:
            return "Reproductive"
        return "Endocrine & Other"

    sys_labels = [map_sys(o) for o in organs]
    sys_order = ["CNS & Nerve", "Cardiovascular", "GI & Metabolic", "Respiratory", "Immune & Blood", "Renal & Urinary", "Reproductive", "Endocrine & Other"]

    # Sort organs by physiological system
    org_df = pd.DataFrame({"organ": organs, "system": sys_labels})
    org_df["sys_rank"] = org_df["system"].map({s: i for i, s in enumerate(sys_order)})
    org_df = org_df.sort_values(["sys_rank", "organ"]).reset_index(drop=True)
    sorted_orgs = org_df["organ"].tolist()

    mat_log = np.log1p(sub[sorted_orgs].values)  # (n_targets, 54)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.4), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.8]})

    # Panel A: System-Level Max/Mean Safety DotPlot across 8 Human Organ Systems
    sys_means = np.zeros((len(sub.index), len(sys_order)))
    sys_maxs = np.zeros((len(sub.index), len(sys_order)))
    for s_i, s_name in enumerate(sys_order):
        s_orgs = org_df[org_df["system"] == s_name]["organ"].tolist()
        vals = np.log1p(sub[s_orgs].values)
        sys_means[:, s_i] = vals.mean(axis=1)
        sys_maxs[:, s_i] = vals.max(axis=1)

    xs, ys, cs, ss = [], [], [], []
    for r_i in range(len(sub.index)):
        for c_i in range(len(sys_order)):
            xs.append(c_i)
            ys.append(r_i)
            cs.append(sys_means[r_i, c_i])
            ss.append(22 + 42 * sys_maxs[r_i, c_i])
    sc_s = axes[0].scatter(xs, ys, c=cs, s=ss, cmap="YlOrRd", vmin=0, vmax=4.5, edgecolors="#222", lw=0.6)
    axes[0].set_xticks(range(len(sys_order)))
    axes[0].set_xticklabels(sys_order, rotation=35, ha="right", fontsize=8.0, fontweight="bold")
    axes[0].set_yticks(range(len(sub.index)))
    axes[0].set_yticklabels(sub.index, fontsize=8.8, fontweight="bold")
    axes[0].set_title("a  Organ-System Safety Burden (8 Systems)", loc="left", fontsize=10.2, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, ls=":", alpha=0.35)
    fig.colorbar(sc_s, ax=axes[0], shrink=0.72, label="Mean log1p(TPM)")

    # Panel B: Full 54-Organ Clustered Heatmap with System Color Bar
    im = axes[1].imshow(mat_log, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4.8)
    axes[1].set_yticks(range(len(sub.index)))
    axes[1].set_yticklabels(sub.index, fontsize=8.8, fontweight="bold")
    # Show representative organ ticks at system boundaries
    short_sys = {
        "CNS & Nerve": "CNS", "Cardiovascular": "Cardio", "GI & Metabolic": "GI/Metab",
        "Respiratory": "Resp", "Immune & Blood": "Immune", "Renal & Urinary": "Renal",
        "Reproductive": "Repro", "Endocrine & Other": "Endocr/Other"
    }
    tick_pos, tick_lbl = [], []
    for s_name in sys_order:
        idxs = org_df.index[org_df["system"] == s_name].tolist()
        if idxs:
            mid = idxs[len(idxs) // 2]
            tick_pos.append(mid)
            tick_lbl.append(f"{short_sys.get(s_name, s_name)} ({len(idxs)})")
            if idxs[0] > 0:
                axes[1].axvline(idxs[0] - 0.5, color="white", lw=1.4)
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(tick_lbl, rotation=38, ha="right", fontsize=7.8, fontweight="bold")
    axes[1].set_title("b  Full 54-Organ GTEx v8 Expression Atlas (Selective vs Broad Targets)", loc="left", fontsize=10.2, fontweight="bold")
    fig.colorbar(im, ax=axes[1], shrink=0.72, label="log1p(Median TPM)")

    fig.suptitle("Human 54-Organ Translational Safety Atlas: Selective Window (GLP1R) vs Broad Kinases",
                 fontsize=12.0, fontweight="bold", y=1.03)
    save_both(fig, "nature_domain_fig4_human_organ_atlas")


def make_downstream_analysis_figure():
    print("[5/5] Building Downstream Volcano, Driver Waterfall & Signed Pathway Lollipop...")
    ad = sc.read_h5ad(os.path.join(DATA_DIR, "Norman2019_hvg2000.h5ad"))
    p_col = ad.obs["perturbation"].astype(str)
    ctrl_X = ad[p_col == "control"].X
    klf_X = ad[p_col == "KLF1"].X
    ctrl_X = ctrl_X.toarray() if hasattr(ctrl_X, "toarray") else np.asarray(ctrl_X)
    klf_X = klf_X.toarray() if hasattr(klf_X, "toarray") else np.asarray(klf_X)

    delta = klf_X.mean(axis=0) - ctrl_X.mean(axis=0)
    _, pvals = ttest_ind(klf_X, ctrl_X, axis=0, equal_var=False)
    pvals = np.clip(np.nan_to_num(pvals, nan=1.0), 1e-300, 1.0)
    neg_log10_p = -np.log10(pvals)
    genes = np.array([str(g) for g in ad.var_names])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1), dpi=300)
    fig.subplots_adjust(wspace=0.45)

    # Panel A: Differential Expression Volcano Plot
    up = (delta > 0.08) & (neg_log10_p > 8)
    dn = (delta < -0.08) & (neg_log10_p > 8)
    ns = ~(up | dn)
    axes[0].scatter(delta[ns], neg_log10_p[ns], s=10, c="#BDC3C7", alpha=0.55, edgecolors="none", label="Not Sig.", rasterized=True)
    axes[0].scatter(delta[up], neg_log10_p[up], s=24, c="#C0392B", alpha=0.88, edgecolors="#111111", lw=0.4, label=f"Up-regulated (n={up.sum()})")
    axes[0].scatter(delta[dn], neg_log10_p[dn], s=24, c="#2B5B84", alpha=0.88, edgecolors="#111111", lw=0.4, label=f"Down-regulated (n={dn.sum()})")
    axes[0].axvline(0.08, color="#7F8C8D", ls="--", lw=1.0)
    axes[0].axvline(-0.08, color="#7F8C8D", ls="--", lw=1.0)
    axes[0].axhline(8, color="#7F8C8D", ls="--", lw=1.0)
    top_up = np.where(up)[0][np.argsort(-delta[up])[:5]] if up.sum() > 0 else []
    top_dn = np.where(dn)[0][np.argsort(delta[dn])[:4]] if dn.sum() > 0 else []
    for idx in list(top_up) + list(top_dn):
        axes[0].text(delta[idx] + (0.015 if delta[idx] > 0 else -0.015), neg_log10_p[idx] + 1.2,
                     genes[idx], fontsize=7.5, fontweight="bold",
                     ha="left" if delta[idx] > 0 else "right",
                     color="#922B21" if delta[idx] > 0 else "#1B4F72")
    axes[0].set_xlim(-0.35, 0.55)
    axes[0].set_ylim(-2, 52)
    axes[0].set_xlabel("Perturbation Δ Expression (KLF1 − Control)", fontsize=9.0, fontweight="bold")
    axes[0].set_ylabel("-log10(P-value)", fontsize=9.0, fontweight="bold")
    axes[0].set_title("a  Downstream DEG Volcano (KLF1 Perturbation)", loc="left", fontsize=10.2, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(fontsize=7.6, frameon=True, loc="upper left")

    # Panel B: Driver vs Passenger Waterfall Ranking (Norman2019_K562)
    rk = pd.read_csv(os.path.join(SRC_DIR, "Fig2_Overexpr_Epistasis_MultiGene", "exp12_unified_driver_passenger_ranking.csv"))
    rk_n = rk[rk["dataset"] == "Norman2019_K562"].sort_values("normalized_driver_score", ascending=False)
    top_d = pd.concat([rk_n.head(8), rk_n.tail(6)]).sort_values("normalized_driver_score", ascending=True)
    role_col = {"Master Driver": "#C0392B", "Co-Driver": "#2B5B84", "Passenger": "#7F8C8D"}
    cols_w = [role_col.get(r, "#7F8C8D") for r in top_d["role"]]
    y_pos = np.arange(len(top_d))
    axes[1].hlines(y_pos, 1.0, top_d["normalized_driver_score"], color=cols_w, lw=2.4)
    axes[1].scatter(top_d["normalized_driver_score"], y_pos, s=75, c=cols_w, edgecolors="#111111", lw=0.8, zorder=4)
    axes[1].axvline(1.0, color="#2C3E50", ls="--", lw=1.2, label="Equal Share (1.0)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_d["target_gene"], fontsize=8.0, fontweight="bold")
    axes[1].set_xlabel("Normalized Driver Attribution Score", fontsize=9.0, fontweight="bold")
    axes[1].set_title("b  Master Driver vs Passenger Waterfall", loc="left", fontsize=10.2, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(True, axis="x", ls=":", alpha=0.4)
    for r_lbl, r_c in role_col.items():
        axes[1].scatter([], [], s=65, c=r_c, edgecolors="#111", label=r_lbl)
    axes[1].legend(fontsize=7.3, frameon=True, loc="lower right")

    # Panel C: Signed Pathway Activation vs Suppression Diverging Lollipop (24h Drug Response)
    cas = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp32b_temporal_pathway_cascade.csv"))
    cas24 = cas[cas["time_hours"] == cas["time_hours"].max()].sort_values("signed_activity", ascending=True)
    y_p = np.arange(len(cas24))
    c_div = ["#C0392B" if v >= 0 else "#2B5B84" for v in cas24["signed_activity"]]
    axes[2].hlines(y_p, 0.0, cas24["signed_activity"], color=c_div, lw=3.0)
    axes[2].scatter(cas24["signed_activity"], y_p, s=110, c=c_div, edgecolors="#111111", lw=0.9, zorder=4)
    axes[2].axvline(0.0, color="#2C3E50", ls="-", lw=1.1)
    axes[2].set_yticks(y_p)
    axes[2].set_yticklabels(cas24["pathway"], fontsize=8.0, fontweight="bold")
    axes[2].set_xlabel("Signed Pathway Activity (24h Post-Drug)", fontsize=9.0, fontweight="bold")
    axes[2].set_title("c  Downstream Pathway Activation vs Arrest", loc="left", fontsize=10.2, fontweight="bold")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].grid(True, axis="x", ls=":", alpha=0.4)

    fig.suptitle("Downstream Biological Interpretation: DEG Volcano, Driver Attribution & Pathway Polarity",
                 fontsize=12.0, fontweight="bold", y=1.03)
    save_both(fig, "nature_domain_fig5_downstream_volcano_gsea_epistasis")


if __name__ == "__main__":
    make_downstream_analysis_figure()
    print("\nDownstream analysis figure generated successfully!")
