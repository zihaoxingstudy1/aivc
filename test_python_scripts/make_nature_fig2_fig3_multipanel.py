# -*- coding: utf-8 -*-
"""
make_nature_fig2_fig3_multipanel.py
Generate diverse, unified-palette Nature-grade multi-panel figures (non-bar advanced charts):
  1. nature_fig2_overexpr_and_epistasis.png/.pdf:
     - Panel a: Lollipop / Dumbbell Gain Chart (Overexpression Unseen CRISPRa)
     - Panel b: Multi-Metric Parallel Slope / Profile Chart (Double-Gene Combinatorial Epistasis)
     - Panel c: Pareto Efficiency Bubble Plot (MSE vs Non-Additive Epistasis r)
  2. nature_fig3_drug_temporal_scaling.png/.pdf:
     - Panel a: Before-vs-After Slopegraph (Same-Modality Tahoe-100M Pretraining on MIXseq 6h & 24h)
     - Panel b: Data Scaling Law Curve with Reference Bands (Leave-Drug-Out 76 Unseen Drugs)
     - Panel c: Annotated Matrix Heatmap (Blind Tumor Masked-Kinase Phosphorylation Recovery)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = r"D:\data\AIVC\test_results"
OUT_FIG2_PNG = os.path.join(RES_DIR, "nature_fig2_overexpr_and_epistasis.png")
OUT_FIG2_PDF = os.path.join(RES_DIR, "nature_fig2_overexpr_and_epistasis.pdf")
OUT_FIG3_PNG = os.path.join(RES_DIR, "nature_fig3_drug_temporal_scaling.png")
OUT_FIG3_PDF = os.path.join(RES_DIR, "nature_fig3_drug_temporal_scaling.pdf")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})

CRIMSON = "#C0392B"
AMBER = "#E67E22"
NAVY = "#2B5B84"
TEAL = "#16A085"
SLATE = "#34495E"
GRAY = "#7F8C8D"


def make_fig2():
    df_oe = pd.read_csv(os.path.join(RES_DIR, "exp37c_overexpression_results.csv"))
    df_cb = pd.read_csv(os.path.join(RES_DIR, "exp39_norman_combo_epistasis_results.csv"))

    fig = plt.figure(figsize=(16.0, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # Panel a: Lollipop / Dumbbell Gain Chart for Gene Overexpression
    ax_a = fig.add_subplot(gs[0, 0])
    metrics = [("mean_pearson", "Mean All-Gene r"), ("top20_pearson", "Top-20 DEG r"), ("dir_acc_top20", "Top-20 DirAcc")]
    y_pos = np.arange(len(metrics))
    ridge_row = df_oe[df_oe["model"] == "Linear_Ridge_ESM2"].iloc[0]
    aivc_row = df_oe[df_oe["model"] == "AIVC_Overexpr_CisOmniPath_Ens5"].iloc[0]
    sing_row = df_oe[df_oe["model"] == "AIVC_Overexpr_CisOmniPath_Single"].iloc[0]

    for i, (col, lbl) in enumerate(metrics):
        v_r, v_s, v_a = float(ridge_row[col]), float(sing_row[col]), float(aivc_row[col])
        ax_a.plot([v_r, v_a], [i, i], color="#CBD5E1", lw=4.0, zorder=1)
        ax_a.scatter([v_r], [i], color=SLATE, s=75, zorder=3, label="Linear-Ridge (ESM-2)" if i == 0 else None)
        ax_a.scatter([v_s], [i], color=AMBER, s=65, marker="D", zorder=4, label="AIVC Overexpr (Single)" if i == 0 else None)
        ax_a.scatter([v_a], [i], color=CRIMSON, s=95, zorder=5, label="AIVC Overexpr (Ens5)" if i == 0 else None)
        ax_a.text(v_a + 0.012, i, f"+{v_a - v_r:.4f}", va="center", fontsize=8.5, fontweight="bold", color=CRIMSON)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([m[1] for m in metrics], fontsize=9.0, fontweight="bold")
    ax_a.set_xlim(0.40, 0.85)
    ax_a.set_xlabel("Score on Unseen Norman2019 CRISPRa (n = 21)", fontsize=9.0, fontweight="bold")
    ax_a.set_title("a  Unseen Gene Overexpression Gains", loc="left", fontsize=10.5, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=7.8, frameon=True)

    # Panel b: Parallel Coordinate / Multi-Metric Profile for Double-Gene Combo Epistasis
    ax_b = fig.add_subplot(gs[0, 1])
    x_cols = ["mean_pearson", "top20_pearson", "dir_acc_top20", "epistasis_r"]
    x_lbls = ["Mean r", "Top-20 r", "DirAcc", "Epistasis r"]
    x_idx = np.arange(len(x_cols))
    c_map = {
        "AIVC_Combo_Epistasis_Ens5": (CRIMSON, 2.8, "o", 7.0),
        "Additive (dA+dB)": (NAVY, 2.0, "s", 5.5),
        "Linear_Ridge": (SLATE, 1.6, "^", 5.5),
        "Static_MLP": (TEAL, 1.4, "D", 4.5),
        "Train_Mean": (GRAY, 1.2, "v", 4.5),
    }
    for _, row in df_cb.iterrows():
        m = row["model"]
        col, lw, mk, ms = c_map.get(m, (GRAY, 1.2, "o", 4))
        vals = [row[c] for c in x_cols]
        ax_b.plot(x_idx, vals, color=col, lw=lw, marker=mk, markersize=ms, label=m.replace("AIVC_Combo_Epistasis_Ens5", "AIVC Epistasis Ens5"))
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(x_lbls, fontsize=9.0, fontweight="bold")
    ax_b.set_ylim(-0.05, 1.06)
    ax_b.set_ylabel("Metric Value (Unseen Combos, n = 27)", fontsize=9.0, fontweight="bold")
    ax_b.set_title("b  Multi-Metric Combo & Epistasis Profile", loc="left", fontsize=10.5, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower left", fontsize=7.5, frameon=True)

    # Panel c: Pareto Efficiency Bubble Plot (MSE vs Epistasis r)
    ax_c = fig.add_subplot(gs[0, 2])
    for _, row in df_cb.iterrows():
        m = row["model"]
        col = c_map.get(m, (GRAY, 1, "o", 4))[0]
        sz = (row["top20_pearson"] ** 3) * 320.0
        ax_c.scatter([row["mse"]], [row["epistasis_r"]], s=sz, color=col, alpha=0.85, edgecolors="#111111", linewidth=1.0, zorder=4)
        short_nm = m.replace("AIVC_Combo_Epistasis_Ens5", "AIVC Ens5\n(MSE -57.3%)")
        offsets = {
            "AIVC_Combo_Epistasis_Ens5": (0.00028, 0.015),
            "Linear_Ridge": (-0.00045, -0.068),
            "Static_MLP": (0.00035, -0.012),
            "Additive (dA+dB)": (0.00035, 0.015),
            "Train_Mean": (-0.0016, 0.032),
        }
        dx, dy = offsets.get(m, (0.0002, 0.02))
        ax_c.text(row["mse"] + dx, row["epistasis_r"] + dy, short_nm, fontsize=7.8,
                  fontweight="bold" if "AIVC" in m else "normal", color=col)
    ax_c.set_xlabel("Prediction MSE (Lower is Better ←)", fontsize=9.0, fontweight="bold")
    ax_c.set_ylabel("Non-Additive Epistasis Residual r (Higher ↑)", fontsize=9.0, fontweight="bold")
    ax_c.set_xlim(0.0005, 0.0078)
    ax_c.set_ylim(-0.06, 0.72)
    ax_c.set_title("c  Pareto Optimality (Bubble = Top-20 r)", loc="left", fontsize=10.5, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Figure 2 | Gene Overexpression Pretraining & Combinatorial Epistasis Modeling (Norman2019)",
                 fontsize=12.2, fontweight="bold", y=1.02)
    plt.savefig(OUT_FIG2_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_FIG2_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Nature Figure 2 -> {OUT_FIG2_PNG}")


def make_fig3():
    df_38 = pd.read_csv(os.path.join(RES_DIR, "exp38_tahoe_drug_temporal_results.csv"))
    df_40 = pd.read_csv(os.path.join(RES_DIR, "exp40b_leave_drug_out_multikernel_results.csv"))
    df_35 = pd.read_csv(os.path.join(RES_DIR, "exp35_masked_kinase_and_kalman_results.csv"))

    fig = plt.figure(figsize=(16.2, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    # Panel a: Paired Slopegraph (Same-Modality Tahoe-100M Pretraining Gains on MIXseq 6h & 24h)
    ax_a = fig.add_subplot(gs[0, 0])
    pairs = [
        ("24h_All13Drugs", "mean_pearson", "24h Mean r", CRIMSON, "-"),
        ("24h_All13Drugs", "top20_pearson", "24h Top-20 r", AMBER, "-"),
        ("24h_All13Drugs", "dir_acc_top20", "24h DirAcc", "#9B59B6", "-"),
        ("6h_Paired", "mean_pearson", "6h Mean r", NAVY, "--"),
        ("6h_Paired", "top20_pearson", "6h Top-20 r", TEAL, "--"),
        ("6h_Paired", "dir_acc_top20", "6h DirAcc", "#27AE60", "--"),
    ]
    for hor, col, lbl, c, ls in pairs:
        v0 = float(df_38[(df_38["horizon"] == hor) & (df_38["model"] == "Exp38_NoPretrain_StructOnly")][col].iloc[0])
        v1 = float(df_38[(df_38["horizon"] == hor) & (df_38["model"] == "Exp38_TahoeDrugPretrained")][col].iloc[0])
        ax_a.plot([0, 1], [v0, v1], color=c, linestyle=ls, lw=2.2, marker="o", markersize=6.5, label=lbl)
        ax_a.text(-0.04, v0, f"{v0:.3f}", ha="right", va="center", fontsize=7.6, color=c)
        ax_a.text(1.04, v1, f"{v1:.3f} (+{v1-v0:.3f})", ha="left", va="center", fontsize=7.6, fontweight="bold", color=c)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["No Pretrain\n(StructOnly)", "Tahoe-100M\nPretrained"], fontsize=9.0, fontweight="bold")
    ax_a.set_xlim(-0.32, 1.55)
    ax_a.set_ylim(0.44, 0.94)
    ax_a.set_ylabel("Unseen-Cell-Line Performance (MIXseq, n = 22)", fontsize=9.0, fontweight="bold")
    ax_a.set_title("a  Same-Modality Drug Pretraining Slopegraph", loc="left", fontsize=10.3, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.legend(loc="center left", bbox_to_anchor=(0.01, 0.56), fontsize=7.0, frameon=True, ncol=2)

    # Panel b: Smooth Scaling Law Curve with Baseline Reference Lines (Leave-Drug-Out 76 Unseen Drugs)
    ax_b = fig.add_subplot(gs[0, 1])
    sc_df = df_40[df_40["model"] == "AIVC_Pathway_Target_SMILES"].sort_values("n_train_drugs")
    tm_top20 = float(df_40[df_40["model"] == "Train_Mean"]["top20_pearson"].iloc[0])
    rg_top20 = float(df_40[df_40["model"] == "Linear_Ridge_SMILES"]["top20_pearson"].iloc[0])
    sm_top20 = float(df_40[df_40["model"] == "SMILES_Kernel_Only"]["top20_pearson"].iloc[0])

    ax_b.plot(sc_df["n_train_drugs"], sc_df["top20_pearson"], color=CRIMSON, lw=2.6, marker="o", markersize=7.5, label="AIVC Pathway+Target+SMILES (Top-20 r)")
    ax_b.fill_between(sc_df["n_train_drugs"], tm_top20, sc_df["top20_pearson"], where=(sc_df["top20_pearson"] >= tm_top20), color=CRIMSON, alpha=0.14)
    ax_b.axhline(sm_top20, color=NAVY, linestyle=":", lw=1.5, label=f"SMILES Kernel Only ({sm_top20:.4f})")
    ax_b.axhline(tm_top20, color=GRAY, linestyle="--", lw=1.4, label=f"Train-Mean ({tm_top20:.4f})")
    ax_b.axhline(rg_top20, color=SLATE, linestyle="-.", lw=1.3, label=f"Linear-Ridge SMILES ({rg_top20:.4f})")
    for _, r in sc_df.iterrows():
        ax_b.text(r["n_train_drugs"], r["top20_pearson"] + 0.004, f"{r['top20_pearson']:.4f}", ha="center", fontsize=7.8, fontweight="bold", color=CRIMSON)
    ax_b.set_xlabel("Number of Training Compounds (Tahoe-100M)", fontsize=9.0, fontweight="bold")
    ax_b.set_ylabel("Zero-Shot Unseen-Drug Top-20 DEG r (n = 76)", fontsize=9.0, fontweight="bold")
    ax_b.set_ylim(0.61, 0.715)
    ax_b.set_title("b  Leave-Drug-Out Monotonic Scaling Law", loc="left", fontsize=10.3, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower right", fontsize=7.5, frameon=True)

    # Panel c: Annotated Heatmap for Blind Masked-Kinase Phosphorylation Recovery
    ax_c = fig.add_subplot(gs[0, 2])
    sub35 = df_35[(df_35["part"] == "A_MaskedKinase") & (df_35["subset"] == "Masked_Kinase_Phospho")]
    piv = sub35.pivot(index="cohort", columns="model", values="median_r")
    col_ord = ["Cis_mRNA_MaskedKinase", "Exp28_5Omics_MaskedKinase", "Exp28+OmniPathCausal_MaskedKinase", "Exp28_5Omics_Unmasked"]
    row_ord = ["TCGA-LUAD", "TCGA-OV", "CPTAC-LUAD", "CPTAC-OV"]
    piv = piv.reindex(index=row_ord, columns=col_ord)
    mat = piv.values.astype(float)
    im = ax_c.imshow(mat, cmap="RdYlBu_r", vmin=-0.05, vmax=0.32, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            tc = "#FFFFFF" if v > 0.22 else "#111111"
            ax_c.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=tc)
    ax_c.set_xticks(np.arange(len(col_ord)))
    ax_c.set_xticklabels(["Cis-mRNA\n(Masked)", "5-Omics\n(Masked)", "5-Omics+\nOmniPath\n(Masked)", "5-Omics\n(Unmasked)"], fontsize=7.8)
    ax_c.set_yticks(np.arange(len(row_ord)))
    ax_c.set_yticklabels(row_ord, fontsize=8.8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cb.set_label("Median Pearson r", fontsize=8.2)
    ax_c.set_title("c  Blind Tumor Masked-Kinase Phospho Matrix", loc="left", fontsize=10.3, fontweight="bold")

    fig.suptitle("Figure 3 | Pharmacological Pretraining, Zero-Shot Chemical Scaling & Masked-Kinase Recovery",
                 fontsize=12.2, fontweight="bold", y=1.02)
    plt.savefig(OUT_FIG3_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_FIG3_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Nature Figure 3 -> {OUT_FIG3_PNG}")


if __name__ == "__main__":
    make_fig2()
    make_fig3()
