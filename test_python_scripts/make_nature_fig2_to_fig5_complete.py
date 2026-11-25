# -*- coding: utf-8 -*-
"""
make_nature_fig2_to_fig5_complete.py
Generate unified Nature-grade 300-DPI PNG + Vector PDF multi-panel figures
for Figure 2 (8/9-model Norman Overexpression & Combo Epistasis),
Figure 4 (Temporal Pathway Cascade + 18-Day State-Anchored ODE + LINCS 1,794-Compound Scaling Law),
and Figure 5 (Clinical Cohorts + Resistant Breast Cancer Pareto Window + GTEx/EyeGEx Safety Guard).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = r"D:\data\AIVC\source_data"
OUT_DIR = r"D:\data\AIVC\test_results"

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
PURPLE = "#8E44AD"
SLATE = "#34495E"
GRAY = "#7F8C8D"


def make_fig2():
    df_s = pd.read_csv(os.path.join(SRC_DIR, "Fig2_Overexpr_Epistasis_MultiGene", "sota_benchmark_norman_overexpr.csv"))
    df_c = pd.read_csv(os.path.join(SRC_DIR, "Fig2_Overexpr_Epistasis_MultiGene", "sota_benchmark_norman_combo.csv"))

    fig = plt.figure(figsize=(16.5, 5.6), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.33)

    # Panel a: 8-Model Overexpression Horizontal Dot-Lollipop Plot
    ax_a = fig.add_subplot(gs[0, 0])
    df_s_ord = df_s.sort_values("top20_pearson", ascending=True).reset_index(drop=True)
    for i, r in df_s_ord.iterrows():
        c = CRIMSON if "Ens5" in r["model"] else (AMBER if "Single" in r["model"] else (SLATE if "Ridge" in r["model"] else NAVY))
        ax_a.plot([0.10, r["top20_pearson"]], [i, i], color="#CBD5E1", lw=1.8, zorder=1)
        ax_a.scatter([r["top20_pearson"]], [i], color=c, s=80 if "AIVC" in r["model"] else 55, zorder=3, label="Top-20 DEG r" if i == 0 else None)
        ax_a.scatter([r["mean_pearson"]], [i], color=c, marker="s", s=38, alpha=0.65, zorder=3, label="Mean All-Gene r" if i == 0 else None)
        ax_a.text(r["top20_pearson"] + 0.012, i, f"{r['top20_pearson']:.4f}", va="center", fontsize=7.6,
                  fontweight="bold" if "AIVC" in r["model"] else "normal", color=c)
    ax_a.set_yticks(np.arange(len(df_s_ord)))
    lbls_s = [m.replace("AIVC_Overexpr_CisOmniPath_", "AIVC Overexpr ").replace("_", "-") for m in df_s_ord["model"]]
    ax_a.set_yticklabels(lbls_s, fontsize=8.0)
    ax_a.set_xlim(0.10, 0.68)
    ax_a.set_xlabel("Correlation (Unseen Norman2019 CRISPRa, n = 21)", fontsize=8.8, fontweight="bold")
    ax_a.set_title("a  8-Model Overexpression SOTA Benchmark", loc="left", fontsize=10.3, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=7.6, frameon=True)

    # Panel b: 9-Model Double-Gene Combo Parallel Metric Profile
    ax_b = fig.add_subplot(gs[0, 1])
    x_cols = ["mean_pearson", "top20_pearson", "dir_acc_top20", "epistasis_r"]
    x_lbls = ["Mean r", "Top-20 r", "DirAcc", "Epistasis r"]
    x_idx = np.arange(len(x_cols))
    style_map = {
        "AIVC_Combo_Epistasis_Ens5": (CRIMSON, 2.8, "o", 7.0, "AIVC Epistasis Ens5"),
        "Additive (dA+dB)": (NAVY, 2.0, "s", 5.5, "Additive (dA+dB)"),
        "Linear_Ridge": (SLATE, 1.6, "^", 5.0, "Linear-Ridge"),
        "scGPT (WholeHuman)": (TEAL, 1.6, "D", 4.8, "scGPT"),
        "Geneformer-V2 (104M)": (PURPLE, 1.4, "P", 4.8, "Geneformer-V2"),
        "scFoundation (100M)": ("#2980B9", 1.3, "v", 4.5, "scFoundation"),
        "GEARS (SGConv)": ("#95A5A6", 1.2, "X", 4.5, "GEARS-SGConv"),
    }
    for _, r in df_c.iterrows():
        m = r["model"]
        if m not in style_map:
            continue
        col, lw, mk, ms, disp = style_map[m]
        ax_b.plot(x_idx, [r[c] for c in x_cols], color=col, lw=lw, marker=mk, markersize=ms, label=disp)
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(x_lbls, fontsize=8.8, fontweight="bold")
    ax_b.set_ylim(-0.05, 1.06)
    ax_b.set_ylabel("Score (Unseen Double-Gene Combos, n = 27)", fontsize=8.8, fontweight="bold")
    ax_b.set_title("b  9-Model Combo & Epistasis Profile", loc="left", fontsize=10.3, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower left", fontsize=7.2, frameon=True, ncol=2)

    # Panel c: 9-Model Pareto Bubble Plot (MSE vs Epistasis r)
    ax_c = fig.add_subplot(gs[0, 2])
    offsets = {
        "AIVC_Combo_Epistasis_Ens5": (0.0005, 0.035),
        "Additive (dA+dB)": (0.0006, 0.012),
        "Linear_Ridge": (-0.0018, -0.055),
        "Static_MLP": (0.0005, 0.032),
        "scGPT (WholeHuman)": (0.0006, -0.015),
        "Geneformer-V2 (104M)": (-0.0038, -0.035),
        "Train_Mean": (0.0006, 0.012),
        "GEARS (SGConv)": (0.0006, -0.025),
        "scFoundation (100M)": (-0.0042, 0.032),
    }
    for _, r in df_c.iterrows():
        m = r["model"]
        col = style_map[m][0] if m in style_map else GRAY
        sz = (r["top20_pearson"] ** 3) * 280.0
        ax_c.scatter([r["mse"]], [r["epistasis_r"]], s=sz, color=col, alpha=0.85, edgecolors="#111111", lw=0.9, zorder=4)
        short = m.replace("AIVC_Combo_Epistasis_Ens5", "AIVC Ens5\n(MSE -57.3%)").replace(" (WholeHuman)", "").replace(" (104M)", "").replace(" (100M)", "").replace(" (SGConv)", "")
        dx, dy = offsets.get(m, (0.0003, 0.02))
        ax_c.text(r["mse"] + dx, r["epistasis_r"] + dy, short, fontsize=7.2,
                  fontweight="bold" if "AIVC" in m else "normal", color=col)
    ax_c.set_xlabel("Prediction MSE (Lower is Better ←)", fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Non-Additive Epistasis r (Higher ↑)", fontsize=8.8, fontweight="bold")
    ax_c.set_xlim(0.0, 0.0175)
    ax_c.set_ylim(-0.07, 0.72)
    ax_c.set_title("c  9-Model Pareto Frontier (Bubble = Top-20 r)", loc="left", fontsize=10.3, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Figure 2 | Gene Overexpression & Double-Gene Epistasis vs. Open-Source SOTA Models (Norman2019)",
                 fontsize=12.0, fontweight="bold", y=1.02)
    for ext in ["png", "pdf"]:
        p = os.path.join(OUT_DIR, f"nature_fig2_overexpr_and_epistasis.{ext}")
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved updated Nature Figure 2")


def make_fig4():
    df_cas = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp32b_temporal_pathway_cascade.csv"))
    df_41 = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp41_schiebinger_18day_ode_results.csv"))
    df_41b = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp41b_schiebinger_anchored_ode_results.csv"))
    df_42 = pd.read_csv(os.path.join(SRC_DIR, "Fig3_Drug_Temporal_Scaling_Phospho", "exp42_lincs_l1000_mega_scaling_results.csv"))

    fig = plt.figure(figsize=(16.8, 5.4), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.40)

    # Panel a: Continuous Temporal Pathway Cascade (0h -> 48h)
    ax_a = fig.add_subplot(gs[0, 0])
    t_cols = [c for c in df_cas.columns if c.startswith("t_")]
    if t_cols and "pathway" in df_cas.columns:
        hrs = [float(c.replace("t_", "").replace("h", "")) for c in t_cols]
        top_pws = df_cas.head(5)
        p_cols = [CRIMSON, NAVY, TEAL, AMBER, PURPLE]
        for idx, (_, r) in enumerate(top_pws.iterrows()):
            ax_a.plot(hrs, [r[c] for c in t_cols], marker="o", lw=2.1, color=p_cols[idx % len(p_cols)], label=str(r["pathway"])[:26])
    elif "time_hours" in df_cas.columns:
        p_cols = [CRIMSON, NAVY, TEAL, AMBER, PURPLE]
        for idx, (pw, grp) in enumerate(df_cas.groupby("pathway")):
            if idx >= 5:
                break
            g2 = grp.sort_values("time_hours")
            ax_a.plot(g2["time_hours"], g2["signed_activity"], marker="o", lw=2.1, color=p_cols[idx], label=str(pw)[:26])
    ax_a.axhline(0, color="#94A3B8", linestyle="--", lw=0.9)
    ax_a.set_xlabel("Time Post-Perturbation (Hours)", fontsize=8.8, fontweight="bold")
    ax_a.set_ylabel("Predicted Pathway Activity (Signed Delta)", fontsize=8.8, fontweight="bold")
    ax_a.set_title("a  0h → 48h Grey-Box ODE Pathway Cascade", loc="left", fontsize=10.3, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.legend(loc="best", fontsize=7.0, frameon=True)

    # Panel b: 18-Day Reprogramming Extrapolation (Open-Loop vs State-Anchored ODE)
    ax_b = fig.add_subplot(gs[0, 1])
    m_rows = [
        ("Static MLP(t)", df_41[df_41["model"] == "Static_MLP_Time"].iloc[0], GRAY),
        ("Open-Loop ODE (D0)", df_41[df_41["model"] == "AIVC_FastSlow_ODE_Ens5"].iloc[0], AMBER),
        ("Linear Extrap", df_41b[df_41b["model"] == "Linear_Time_Extrap"].iloc[0], SLATE),
        ("Persistence (D12)", df_41b[df_41b["model"] == "Persistence_Day12"].iloc[0], NAVY),
        ("State-Anchored ODE", df_41b[df_41b["model"] == "AIVC_StateAnchored_ODE_Ens5"].iloc[0], CRIMSON),
    ]
    for i, (nm, row, c) in enumerate(m_rows):
        ax_b.plot([0.55, row["mean_pearson"]], [i, i], color="#CBD5E1", lw=2.5, zorder=1)
        ax_b.scatter([row["mean_pearson"]], [i], color=c, s=85, zorder=3, label="Mean r" if i == 0 else None)
        ax_b.scatter([row["top20_pearson"]], [i], color=c, marker="D", s=55, alpha=0.8, zorder=3, label="Top-20 r" if i == 0 else None)
        ax_b.text(max(row["mean_pearson"], row["top20_pearson"]) + 0.012, i, f"{row['mean_pearson']:.4f}", va="center", fontsize=8.0,
                  fontweight="bold" if "Anchored" in nm else "normal", color=c)
    ax_b.set_yticks(np.arange(len(m_rows)))
    ax_b.set_yticklabels([m[0] for m in m_rows], fontsize=8.5, fontweight="bold")
    ax_b.set_xlim(0.55, 0.96)
    ax_b.set_xlabel("Correlation on Unseen Future D12.5–D18", fontsize=8.8, fontweight="bold")
    ax_b.set_title("b  18-Day Reprogramming ODE Extrapolation", loc="left", fontsize=10.3, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel c: LINCS L1000 1,794-Compound Monotonic MSE Scaling Law
    ax_c = fig.add_subplot(gs[0, 2])
    sc42 = df_42[df_42["model"] == "AIVC_MultiKernel_Chem"].sort_values("n_train_drugs")
    mse_ridge = float(df_42[df_42["model"] == "Linear_Ridge_SMILES"]["mse"].iloc[0])
    mse_mean = float(df_42[df_42["model"] == "Train_Mean"]["mse"].iloc[0])
    ax_c.plot(sc42["n_train_drugs"], sc42["mse"], color=CRIMSON, lw=2.6, marker="o", markersize=7.0, label="AIVC Multi-Kernel Chem")
    ax_c.fill_between(sc42["n_train_drugs"], sc42["mse"], mse_ridge, where=(sc42["mse"] <= mse_ridge), color=CRIMSON, alpha=0.14)
    ax_c.axhline(mse_ridge, color=SLATE, linestyle="-.", lw=1.5, label=f"Linear-Ridge SMILES ({mse_ridge:.4f})")
    ax_c.axhline(mse_mean, color=GRAY, linestyle="--", lw=1.4, label=f"Train-Mean ({mse_mean:.4f})")
    for _, r in sc42.iterrows():
        ax_c.text(r["n_train_drugs"], r["mse"] + 0.004, f"{r['mse']:.4f}", ha="center", fontsize=7.4, fontweight="bold", color=CRIMSON)
    ax_c.set_xlabel("Training Compounds in LINCS L1000 (GSE70138)", fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Unseen-Drug MSE (n = 359, Lower ↓)", fontsize=8.8, fontweight="bold")
    ax_c.set_ylim(0.87, 0.968)
    ax_c.set_title("c  1,794-Compound Leave-Drug-Out Scaling Law", loc="left", fontsize=10.3, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper right", fontsize=7.5, frameon=True)

    fig.suptitle("Figure 4 | Continuous Temporal Cascades, 18-Day Developmental ODE & LINCS L1000 Chemical Scaling",
                 fontsize=12.0, fontweight="bold", y=1.02)
    for ext in ["png", "pdf"]:
        p = os.path.join(OUT_DIR, f"nature_fig4_temporal_and_lincs_scaling.{ext}")
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Nature Figure 4")


def make_fig5():
    df_19 = pd.read_csv(os.path.join(SRC_DIR, "Fig5_Clinical_PanCancer_Safety", "exp19_intrinsic_vs_tme_ablation.csv"))
    df_23 = pd.read_csv(os.path.join(SRC_DIR, "Fig5_Clinical_PanCancer_Safety", "exp23_resistant_patient_golden_combos.csv"))
    df_sf = pd.read_csv(os.path.join(SRC_DIR, "Fig5_Clinical_PanCancer_Safety", "aivc_flagship_safety_report.csv"))

    fig = plt.figure(figsize=(16.5, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel a: Clinical Cohorts Four-Arm Matrix Heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    piv19 = df_19.pivot_table(index="cohort_id", columns="arm", values="metric_value", aggfunc="mean")
    arm_cols = [c for c in ["Arm1_Intrinsic_BioMolKG", "Arm2_TME_Only", "Arm3_Transcriptome_2000g", "Arm5_Full_Fusion_2000g"] if c in piv19.columns]
    if not arm_cols:
        arm_cols = list(piv19.columns[:4])
    piv19 = piv19[arm_cols].dropna().head(8)
    im = ax_a.imshow(piv19.values, cmap="YlGnBu", vmin=0.45, vmax=0.90, aspect="auto")
    for i in range(piv19.shape[0]):
        for j in range(piv19.shape[1]):
            v = piv19.values[i, j]
            ax_a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#FFFFFF" if v > 0.72 else "#111111")
    ax_a.set_xticks(np.arange(len(arm_cols)))
    ax_a.set_xticklabels([c.replace("Arm1_Intrinsic_BioMolKG", "Intrinsic\nBioMolKG").replace("Arm2_TME_Only", "TME\nOnly").replace("Arm3_Transcriptome_2000g", "Full\nRNA").replace("Arm5_Full_Fusion_2000g", "Full\nFusion") for c in arm_cols], fontsize=7.8)
    ax_a.set_yticks(np.arange(len(piv19.index)))
    ax_a.set_yticklabels(piv19.index, fontsize=8.0, fontweight="bold")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.set_label("ROC-AUC / C-index", fontsize=8.0)
    ax_a.set_title("a  Clinical Treatment Cohorts (Four-Arm Ablation)", loc="left", fontsize=10.0, fontweight="bold")

    # Panel b: Resistant Breast Cancer 1,770-Combo Pareto Scatter
    ax_b = fig.add_subplot(gs[0, 1])
    sc = ax_b.scatter(
        df_23["vital_organ_max_bliss"], df_23["rd_patient_synergy_bliss"],
        c=df_23["therapeutic_window_score"], cmap="RdYlGn", s=16, alpha=0.65, edgecolors="none"
    )
    top3 = df_23.sort_values("therapeutic_window_score", ascending=False).head(3)
    ax_b.scatter(top3["vital_organ_max_bliss"], top3["rd_patient_synergy_bliss"], s=65, facecolors="none", edgecolors=CRIMSON, lw=1.6, label="Top Pareto Combos")
    cb2 = fig.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.04)
    cb2.set_label("Therapeutic Window Score", fontsize=8.0)
    ax_b.set_xlabel("Worst Vital Organ Toxicity (Max Bliss)", fontsize=8.8, fontweight="bold")
    ax_b.set_ylabel("Resistant Tumor Synergy (Bliss)", fontsize=8.8, fontweight="bold")
    ax_b.set_title("b  531 Resistant BRCA: 1,770-Combo Pareto Window", loc="left", fontsize=10.0, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel c: GTEx 54-Organ + EyeGEx 453-Retina Multi-Target Safety Matrix
    ax_c = fig.add_subplot(gs[0, 2])
    y_g = np.arange(len(df_sf))
    ax_c.hlines(y_g, 0, np.log1p(df_sf["eyegex_retina_median"]), color=CRIMSON, lw=2.5, alpha=0.7)
    ax_c.scatter(np.log1p(df_sf["eyegex_retina_median"]), y_g, color=CRIMSON, s=85, label="Human Retina log1p(CPM)", zorder=4)
    ax_c.scatter(np.log1p(df_sf["gtex_top_tpm"]), y_g, color=NAVY, marker="s", s=70, label="GTEx Peak Organ log1p(TPM)", zorder=4)
    for i, r in df_sf.iterrows():
        ax_c.text(max(np.log1p(r["eyegex_retina_median"]), np.log1p(r["gtex_top_tpm"])) + 0.14, i,
                  f"{int(r['gtex_organs_above_thresh'])}/54 organs", va="center", fontsize=7.4, color=SLATE)
    ax_c.set_yticks(y_g)
    ax_c.set_yticklabels(df_sf["gene"], fontsize=8.8, fontweight="bold")
    ax_c.set_xlim(-0.2, 6.6)
    ax_c.set_xlabel("Baseline Expression log1p(TPM / CPM)", fontsize=8.8, fontweight="bold")
    ax_c.set_title("c  54-Organ (GTEx) + 453-Retina (EyeGEx) Guard", loc="left", fontsize=10.0, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", linestyle="--", alpha=0.35)
    ax_c.legend(loc="lower right", fontsize=7.4, frameon=True)

    fig.suptitle("Figure 5 | Clinical Cohort Translation, Resistant-Patient Pareto Screening & Multi-Organ/Retina Safety",
                 fontsize=12.0, fontweight="bold", y=1.02)
    for ext in ["png", "pdf"]:
        p = os.path.join(OUT_DIR, f"nature_fig5_clinical_and_safety.{ext}")
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Nature Figure 5")


if __name__ == "__main__":
    make_fig2()
    make_fig4()
    make_fig5()
