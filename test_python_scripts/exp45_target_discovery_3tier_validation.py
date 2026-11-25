# -*- coding: utf-8 -*-
"""
exp45_target_discovery_3tier_validation.py
Three-tier rigorous validation of AIVC Target Discovery:
  1. Cross-Disease Target Applicability across 10 major TCGA cancer cohorts
  2. Classical Approved Oncology Target Recall & Enrichment (Fisher Exact / Hypergeometric test)
  3. Individualized Patient-Specific Target Stratification & Novel Candidate Discovery
     (verified against real per-patient TCGA RPPA phosphoprotein measurements)
Outputs:
  - D:\data\AIVC\test_results\exp45_cross_disease_targets.csv
  - D:\data\AIVC\test_results\exp45_patient_specific_targets.csv
  - D:\data\AIVC\test_results\nature_domain_fig6_target_discovery_3tier.png/.pdf
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, fisher_exact, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_NPZ = r"D:\data\AIVC\data\exp24_multiomics_cache.npz"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
EXP28_W = r"D:\data\AIVC\source_data\Fig4_5Omics_Mechanistic_Discovery\exp28_learned_regulator_weights.csv"
TAHOE_DRUG = r"D:\data\AIVC\data\tahoe_100m\metadata\drug_metadata.parquet"

OUT_DISEASE_CSV = r"D:\data\AIVC\test_results\exp45_cross_disease_targets.csv"
OUT_PATIENT_CSV = r"D:\data\AIVC\test_results\exp45_patient_specific_targets.csv"
OUT_PNG = r"D:\data\AIVC\test_results\nature_domain_fig6_target_discovery_3tier.png"
OUT_PDF = r"D:\data\AIVC\test_results\nature_domain_fig6_target_discovery_3tier.pdf"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=== Exp45: Three-Tier Validation of AIVC Target Discovery ===")
    ck = np.load(CACHE_NPZ, allow_pickle=True)
    rna_genes = [str(g).upper() for g in ck["top_rna"]]
    g2i = {g: i for i, g in enumerate(rna_genes)}
    rppa_abs = [str(a) for a in ck["tcga_rppa_abs"]]
    a2i = {a: i for i, a in enumerate(rppa_abs)}
    X_tcga = ck["X_tcga"].astype(np.float32)
    Y_rppa = ck["Y_tcga_rppa"].astype(np.float32)
    C_tcga = np.array([str(c) for c in ck["C_tcga"]])

    # Load OmniPath directed signaling edges
    edges = pd.read_parquet(os.path.join(PRIORS_DIR, "directed_signaling_edges.parquet"))
    scol, tcol, sgcol = edges.columns[0], edges.columns[1], edges.columns[2]
    edges[scol] = edges[scol].astype(str).str.upper()
    edges[tcol] = edges[tcol].astype(str).str.upper()

    # Build normalized causal footprint matrix F (n_regulators x n_rna_genes) for key druggable & novel hubs
    candidate_hubs = [
        "EGFR", "ERBB2", "BRAF", "MAP2K1", "AKT1", "MTOR", "MET", "SRC", "STAT3",
        "KIT", "CDK4", "YAP1", "PTPN11", "CDKN2A", "PIK3CA", "RAF1"
    ]
    F_mat = np.zeros((len(candidate_hubs), len(rna_genes)), dtype=np.float32)
    hub_edge_counts = {}
    for hi, hub in enumerate(candidate_hubs):
        sub_e = edges[edges[scol] == hub]
        hub_edge_counts[hub] = len(sub_e)
        cnt = 0
        for _, r in sub_e.iterrows():
            ti = g2i.get(r[tcol], -1)
            if ti >= 0:
                F_mat[hi, ti] += float(r[sgcol])
                cnt += 1
        norm = np.linalg.norm(F_mat[hi])
        if norm > 1e-6:
            F_mat[hi] /= norm

    # Per-patient causal target activation score (7359 x n_hubs) = Z-scored X_tcga @ F_mat.T
    X_z = (X_tcga - X_tcga.mean(axis=0, keepdims=True)) / (X_tcga.std(axis=0, keepdims=True) + 1e-6)
    patient_hub_scores = X_z @ F_mat.T  # (7359, len(candidate_hubs))

    # ------------------------------------------------------------------
    # Tier 1: Cross-Disease Applicability across 10 major TCGA cancers
    # ------------------------------------------------------------------
    cancers = ["LUAD", "BRCA", "OV", "COAD", "SKCM", "HNSC", "BLCA", "STAD", "LUSC", "KIRC"]
    hub_to_rppa = {
        "EGFR": "EGFR_pY1068",
        "ERBB2": "HER2_pY1248",
        "AKT1": "AKT_pS473",
        "MTOR": "MTOR_pS2448",
        "STAT3": "STAT3_pY705",
        "SRC": "SRC_pY527",
        "YAP1": "YAP_pS127",
        "MET": "CMET_pY1235",
    }
    dis_rows = []
    for c_name in cancers:
        cmask = (C_tcga == c_name)
        for hub, rppa_col in hub_to_rppa.items():
            if rppa_col not in a2i or hub not in candidate_hubs:
                continue
            hi = candidate_hubs.index(hub)
            ri = a2i[rppa_col]
            s_vec = patient_hub_scores[cmask, hi]
            y_vec = Y_rppa[cmask, ri]
            valid = ~np.isnan(y_vec) & ~np.isnan(s_vec)
            if valid.sum() >= 30 and np.std(s_vec[valid]) > 1e-6 and np.std(y_vec[valid]) > 1e-6:
                r_val, p_val = pearsonr(s_vec[valid], y_vec[valid])
                dis_rows.append({
                    "Cancer": c_name, "N_Patients": int(valid.sum()),
                    "Target_Hub": hub, "RPPA_Phosphosite": rppa_col,
                    "Pearson_r": float(r_val), "P_value": float(p_val)
                })
    df_dis = pd.DataFrame(dis_rows)
    df_dis.to_csv(OUT_DISEASE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Tier 1] Cross-Disease Validation across {len(cancers)} cancers ({len(df_dis)} pairs):")
    piv_d = df_dis.pivot(index="Target_Hub", columns="Cancer", values="Pearson_r")
    print(piv_d.round(3).to_string())

    # ------------------------------------------------------------------
    # Tier 2: Classical Oncology Target Enrichment (Tahoe-100M Approved Targets)
    # ------------------------------------------------------------------
    dm = pd.read_parquet(TAHOE_DRUG)
    app_dm = dm[dm["human-approved"].astype(str).str.lower() == "yes"]
    approved_targets = set()
    for tg_str in app_dm["targets"].dropna():
        for t in str(tg_str).replace(";", ",").split(","):
            t = t.strip().upper()
            if t and t != "NONE":
                approved_targets.add(t)

    w_df = pd.read_csv(EXP28_W)
    rna_w = w_df[w_df["regulator_type"] == "RNA_Gene"].copy()
    rna_w["name_up"] = rna_w["name"].astype(str).str.upper()
    rna_w = rna_w.sort_values("learned_weight", ascending=False).reset_index(drop=True)
    rna_w["is_approved_target"] = rna_w["name_up"].isin(approved_targets)

    # Fisher exact test at Top-300 regulators vs rest of 2,893 genes
    top_k = 300
    a = int(rna_w.iloc[:top_k]["is_approved_target"].sum())
    b = top_k - a
    c = int(rna_w.iloc[top_k:]["is_approved_target"].sum())
    d = len(rna_w) - top_k - c
    odds, p_enrich = fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f"\n[Tier 2] Classical Approved Target Enrichment (Top-{top_k} of {len(rna_w)} genes):")
    print(f"  Approved targets in Top-{top_k}: {a}/{top_k} vs background {c}/{c+d} | Odds Ratio = {odds:.2f} | Fisher p = {p_enrich:.3e}")
    top_hits = rna_w[rna_w["is_approved_target"]].head(12)[["name_up", "learned_weight"]]
    print("  Top Classical Targets Recovered:", [(r["name_up"], round(r["learned_weight"], 3)) for _, r in top_hits.iterrows()])

    # ------------------------------------------------------------------
    # Tier 3: Individualized Patient-Specific Target & Novel Discovery
    # ------------------------------------------------------------------
    # Compare high vs low AIVC causal score patients across distinct cancers
    pat_rows = []
    for c_name, hub, rppa_col in [
        ("BLCA", "STAT3", "STAT3_pY705"),
        ("BRCA", "STAT3", "STAT3_pY705"),
        ("KIRC", "ERBB2", "HER2_pY1248"),
        ("BRCA", "SRC", "SRC_pY527"),
        ("SKCM", "STAT3", "STAT3_pY705"),
        ("LUAD", "YAP1", "YAP_pS127"),
    ]:
        cmask = (C_tcga == c_name)
        hi = candidate_hubs.index(hub)
        ri = a2i[rppa_col]
        sc_v = patient_hub_scores[cmask, hi]
        yp_v = Y_rppa[cmask, ri]
        valid = ~np.isnan(yp_v) & ~np.isnan(sc_v)
        sc_v, yp_v = sc_v[valid], yp_v[valid]
        q25, q75 = np.quantile(sc_v, [0.25, 0.75])
        hi_grp = yp_v[sc_v >= q75]
        lo_grp = yp_v[sc_v <= q25]
        _, p_mw = mannwhitneyu(hi_grp, lo_grp, alternative="greater")
        pat_rows.append({
            "Cancer": c_name, "Target": hub, "RPPA_Phosphosite": rppa_col,
            "Q4_High_Mean_Phospho": float(hi_grp.mean()),
            "Q1_Low_Mean_Phospho": float(lo_grp.mean()),
            "Delta_Phospho": float(hi_grp.mean() - lo_grp.mean()),
            "MannWhitney_p": float(p_mw),
            "N_Per_Quartile": int(len(hi_grp)),
        })
    df_pat = pd.DataFrame(pat_rows)
    df_pat.to_csv(OUT_PATIENT_CSV, index=False, encoding="utf-8-sig")
    print("\n[Tier 3] Individualized Patient Stratification (Top-25% vs Bottom-25% AIVC Causal Score):")
    print(df_pat.to_string(index=False))

    # ------------------------------------------------------------------
    # Plot 3-Panel Nature Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=300)
    fig.subplots_adjust(wspace=0.44)

    # Panel A: Cross-Disease Applicability Heatmap (8 Targets x 10 Cancers)
    im = axes[0].imshow(piv_d.values, cmap="RdBu_r", vmin=-0.35, vmax=0.45, aspect="auto")
    axes[0].set_xticks(range(len(piv_d.columns)))
    axes[0].set_xticklabels(piv_d.columns, rotation=38, ha="right", fontsize=8.0, fontweight="bold")
    axes[0].set_yticks(range(len(piv_d.index)))
    axes[0].set_yticklabels([f"{h}\n({hub_to_rppa[h]})" for h in piv_d.index], fontsize=7.6, fontweight="bold")
    for i in range(piv_d.shape[0]):
        for j in range(piv_d.shape[1]):
            v = piv_d.values[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=6.8, fontweight="bold", color="white" if abs(v) > 0.25 else "#111111")
    axes[0].set_title("a  Cross-Disease Target Phospho Correlation (10 Cancers)", loc="left", fontsize=10.0, fontweight="bold")
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="Causal Score vs RPPA r")

    # Panel B: Classical Approved Target Enrichment & Novel Hubs (Lollipop)
    show_genes = list(top_hits.head(8)["name_up"]) + ["EREG", "YAP1", "CDKN2A", "CDKN2B"]
    sub_w = rna_w[rna_w["name_up"].isin(show_genes)].drop_duplicates("name_up").sort_values("learned_weight", ascending=True)
    y_p = np.arange(len(sub_w))
    c_b = ["#2B5B84" if is_a else "#16A085" for is_a in sub_w["is_approved_target"]]
    axes[1].hlines(y_p, 0, sub_w["learned_weight"], color=c_b, lw=2.6)
    axes[1].scatter(sub_w["learned_weight"], y_p, s=90, c=c_b, edgecolors="#111111", lw=0.8, zorder=4)
    axes[1].scatter([], [], s=75, c="#2B5B84", edgecolors="#111", label="FDA/Approved Target")
    axes[1].scatter([], [], s=75, c="#16A085", edgecolors="#111", label="Novel / Non-Canonical Hub")
    axes[1].set_yticks(y_p)
    axes[1].set_yticklabels(sub_w["name_up"], fontsize=8.2, fontweight="bold")
    axes[1].set_xlabel("AIVC L1-Sparse Causal Importance Weight", fontsize=9.0, fontweight="bold")
    axes[1].set_title(f"b  Target Enrichment (OR={odds:.2f}, p={p_enrich:.1e})", loc="left", fontsize=9.8, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(True, axis="x", ls=":", alpha=0.4)
    axes[1].legend(fontsize=7.6, frameon=True, loc="lower right")

    # Panel C: Individualized Patient Stratification Dumbbell Plot (Q1 Low -> Q4 High)
    lbls_c = [f"{r['Cancer']}: {r['Target']}\n({r['RPPA_Phosphosite']})" for _, r in df_pat.iterrows()]
    y_c = np.arange(len(df_pat))
    for i, r in df_pat.iterrows():
        q1, q4 = r["Q1_Low_Mean_Phospho"], r["Q4_High_Mean_Phospho"]
        axes[2].plot([q1, q4], [i, i], color="#2C3E50", lw=2.2, zorder=2)
        stars = "***" if r["MannWhitney_p"] < 1e-3 else ("**" if r["MannWhitney_p"] < 1e-2 else ("*" if r["MannWhitney_p"] < 0.05 else "n.s."))
        axes[2].text(max(q1, q4) + 0.03, i, f"+{r['Delta_Phospho']:.2f} ({stars})", va="center", ha="left",
                     fontsize=7.8, fontweight="bold", color="#C0392B")
    axes[2].scatter(df_pat["Q1_Low_Mean_Phospho"], y_c, s=95, c="#7F8C8D", edgecolors="#111", lw=0.9, zorder=4, label="Bottom 25% Patients")
    axes[2].scatter(df_pat["Q4_High_Mean_Phospho"], y_c, s=105, c="#C0392B", edgecolors="#111", lw=0.9, zorder=5, label="Top 25% AIVC Score")
    axes[2].axvline(0, color="#95A5A6", ls="--", lw=0.9)
    axes[2].set_yticks(y_c)
    axes[2].set_yticklabels(lbls_c, fontsize=7.8, fontweight="bold")
    axes[2].set_xlim(df_pat["Q1_Low_Mean_Phospho"].min() - 0.15, df_pat["Q4_High_Mean_Phospho"].max() + 0.55)
    axes[2].set_xlabel("Measured Patient RPPA Phosphorylation", fontsize=9.0, fontweight="bold")
    axes[2].set_title("c  Individualized Patient Stratification", loc="left", fontsize=10.0, fontweight="bold")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].grid(True, axis="x", ls=":", alpha=0.35)
    axes[2].legend(fontsize=7.6, frameon=True, loc="upper left")

    fig.suptitle("Three-Tier Validation of AIVC Target Discovery: Cross-Disease, Canonical Recall & Patient Precision",
                 fontsize=11.8, fontweight="bold", y=1.03)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {OUT_PNG} & {OUT_PDF}")


if __name__ == "__main__":
    main()
