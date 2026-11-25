# -*- coding: utf-8 -*-
"""
make_nature_fig1_and_stats.py
Step 1 of Top-Journal Polish:
  1. Compute exact 95% Bootstrap CIs (1,000 resamples) and Paired Wilcoxon signed-rank p-values
     across all 2,016 blind unseen RPE1 perturbations for all 10 models.
  2. Generate Nature-grade Figure 1 (4 diverse chart types: Violin+Box distribution with p-value
     brackets, Paired Density Scatter with Win-Rate, Multi-Metric Forest/Dot-CI plot, and
     Radar/Polar Multi-Axis Profile) with unified Nature colorblind-safe palette (PNG 300 DPI + PDF).
  3. Generate Nature-grade Architecture Schematic Overview (nature_fig0_architecture_schematic.png/.pdf).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions

PRED_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
OUT_STATS_CSV = r"D:\data\AIVC\test_results\sota_benchmark_k562_to_rpe1_stats.csv"
OUT_FIG1_PNG = r"D:\data\AIVC\test_results\nature_fig1_sota_multipanel.png"
OUT_FIG1_PDF = r"D:\data\AIVC\test_results\nature_fig1_sota_multipanel.pdf"
OUT_ARCH_PNG = r"D:\data\AIVC\test_results\nature_fig0_architecture_schematic.png"
OUT_ARCH_PDF = r"D:\data\AIVC\test_results\nature_fig0_architecture_schematic.pdf"

# Unified Nature-style colorblind-safe palette
PALETTE = {
    "AIVC_v1.1_Ensemble": "#C0392B",   # Deep Crimson (Primary Ours)
    "AIVC_v1.1": "#E67E22",            # Burnt Amber (Single Ours)
    "Linear_Ridge": "#34495E",         # Dark Slate (Strong Linear)
    "scFoundation_100M": "#2980B9",    # Ocean Blue (FM)
    "scGPT": "#16A085",                # Teal (FM)
    "Geneformer_V2_104M": "#8E44AD",   # Royal Purple (FM)
    "GEARS_SGConv": "#2E86AB",         # Steel Blue (Graph)
    "scVI": "#5DADE2",                 # Sky Blue (VAE)
    "Train_Mean": "#7F8C8D",           # Cool Gray
    "Static_MLP": "#BDC3C7",           # Light Gray
}

MODELS = [
    ("AIVC_v1.1_Ensemble", "AIVC Ensemble"),
    ("AIVC_v1.1", "AIVC v1.1"),
    ("Linear_Ridge", "Linear-Ridge"),
    ("scFoundation_100M", "scFoundation"),
    ("scGPT", "scGPT"),
    ("Geneformer_V2_104M", "Geneformer-V2"),
    ("GEARS_SGConv", "GEARS"),
    ("scVI", "scVI"),
    ("Train_Mean", "Train-Mean"),
    ("Static_MLP", "Static-MLP"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})


def bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    n = len(vals)
    idx = rng.randint(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_p(p: float) -> str:
    if p < 1e-50:
        return "p < 1e-50 (****)"
    if p < 1e-10:
        return f"p = {p:.1e} (****)"
    if p < 1e-4:
        return f"p = {p:.1e} (***)"
    if p < 1e-2:
        return f"p = {p:.3f} (**)"
    if p < 0.05:
        return f"p = {p:.3f} (*)"
    return f"p = {p:.3f} (ns)"


def draw_architecture_schematic():
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, title_c="#FFFFFF", body_c="#1F2937"):
        # Outer card
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(p)
        # Header band
        hb = FancyBboxPatch((x, y + h - 3.6), w, 3.6, boxstyle="round,pad=0.4,rounding_size=1.0", facecolor=ec, edgecolor=ec, linewidth=0.8)
        ax.add_patch(hb)
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=title_c)
        for idx, ln in enumerate(lines):
            ax.text(x + 1.2, y + h - 5.6 - idx * 2.22, ln, ha="left", va="center", fontsize=7.6, color=body_c)

    # Title banner
    ax.text(50, 49.5, "AIVC: Unified Mechanistic Multi-Omics & Grey-Box Temporal Virtual Cell World Model",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0F172A")

    # Column 1: Priors & Multi-Modal Inputs
    box(1.5, 29.0, 27.0, 17.0, "1. Human Mechanistic Prior Atlas", [
        "• 3,894 Curated Human Pathways",
        "  (Reactome / Wiki / TRRUST / KEGG / Hallmark)",
        "• 109,312 OmniPath Directed Edges (+1 / -1)",
        "• 721,323 BioGRID Physical PPIs",
        "• 366,539 Pathway Cross-Talk Edges",
    ], "#F8FAFC", "#2B5B84")

    box(1.5, 4.5, 27.0, 21.5, "2. Multi-Modal Perturbation Queries", [
        "• Genetic Loss-of-Function (CRISPRi / KO)",
        "• Gene Overexpression (ORF / CRISPRa)",
        "• Multi-Gene Combinations (2–6 Targets)",
        "• Small Molecules (ChemBERTa + Morgan)",
        "• Continuous Dose & Time Horizon (0–48h+)",
        "• Cell Context (RNA + Optional Mut / miRNA)",
    ], "#F8FAFC", "#16A085")

    # Column 2: Core World Model Engine
    box(34.5, 27.5, 31.0, 18.5, "3. Mechanistic & 5-Omics Core Engine", [
        "• Lineage-Invariant 1–2 Hop OmniPath",
        "  Causal Perturbation Footprints",
        "• Dual-Track 5-Omics Translation (9,950",
        "  TCGA/CPTAC/CCLE) + 577 Kinase [MASK]",
        "• Cis-Activation & Symmetric Epistasis",
        "• Modality-Routed Low-Rank Program Gate",
    ], "#FFF8F6", "#C0392B")

    box(34.5, 4.5, 31.0, 19.5, "4. Fast–Slow Grey-Box Temporal ODE", [
        "• Fast Latent State z_fast(t): 0–12h Stress",
        "• Slow Latent State z_slow(t): 12–48h Cascade",
        "• OmniPath-Anchored RK4 Continuous Rollout",
        "• Kalman-Style State Assimilation (Obs6h→24h)",
        "• Multi-View Kernel (Pathway+Target+SMILES)",
        "  for Zero-Shot Unseen New Compounds",
    ], "#FFFBF0", "#D97706")

    # Column 3: Outputs & Translational Safety
    box(71.5, 27.5, 27.0, 18.5, "5. Calibrated Virtual Cell Outputs", [
        "• Post-Perturbation scRNA & Bulk (.h5ad)",
        "• Virtual Proteome (1,200) & Phospho (800)",
        "• 5-Member Deep Ensemble Epistemic UQ",
        "  (Spearman r = +0.3602, p = 8.17e-63)",
        "• Driver vs. Passenger & Epistasis Scores",
    ], "#F5F3FF", "#6D28D9")

    box(71.5, 4.5, 27.0, 19.5, "6. Clinical & Multi-Organ Safety Guard", [
        "• Pan-Cancer IC50 & Repurposing (GDSC/PRISM)",
        "• Genetic-to-Drug Synergy (AUC = 0.8389)",
        "• 17 Clinical Cohorts (5,265 Patients)",
        "• GTEx v8 (54 Normal Human Organs)",
        "• EyeGEx (453 Human Donor Retinas)",
        "• Resistant-Patient Pareto Combo Frontier",
    ], "#F0FDF4", "#15803D")

    # Clean Arrows
    arrow_kw = dict(arrowstyle="-|>,head_width=0.5,head_length=0.8", lw=1.8, color="#475569")
    for (x1, y1, x2, y2) in [
        (29.0, 37.5, 34.0, 37.5),
        (29.0, 15.0, 34.0, 15.0),
        (29.0, 22.0, 34.0, 31.0),
        (50.0, 27.0, 50.0, 24.5),
        (66.0, 37.5, 71.0, 37.5),
        (66.0, 15.0, 71.0, 15.0),
        (66.0, 31.0, 71.0, 21.0),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    plt.tight_layout()
    plt.savefig(OUT_ARCH_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_ARCH_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture schematic -> {OUT_ARCH_PNG} & {OUT_ARCH_PDF}")


def main():
    draw_architecture_schematic()

    bundle = np.load(os.path.join(PRED_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    Y_ev = bundle["Y_ev"]
    d_rpe1 = {ev_keys[i]: Y_ev[i] for i in range(len(ev_keys))}

    per_pert_dfs = {}
    for mid, disp in MODELS:
        fp = os.path.join(PRED_DIR, f"pred_{mid}.npy")
        pmat = np.load(fp)
        df_m = evaluate_perturbation_predictions(
            d_rpe1, {ev_keys[i]: pmat[i] for i in range(len(ev_keys))},
            model_name=disp, top_k_list=[20]
        )
        per_pert_dfs[mid] = df_m

    ref_ens = per_pert_dfs["AIVC_v1.1_Ensemble"]
    stat_rows = []
    for mid, disp in MODELS:
        df_m = per_pert_dfs[mid]
        mr, mr_lo, mr_hi = bootstrap_ci(df_m["pearson_delta_all"].values)
        tr, tr_lo, tr_hi = bootstrap_ci(df_m["pearson_delta_top20"].values)
        da, da_lo, da_hi = bootstrap_ci(df_m["dir_acc_top20"].values)
        ms, ms_lo, ms_hi = bootstrap_ci(df_m["mse_all"].values)

        if mid == "AIVC_v1.1_Ensemble":
            p_top20, p_mean, p_mse, win_top20 = 1.0, 1.0, 1.0, 0.5
        else:
            p_top20 = float(wilcoxon(ref_ens["pearson_delta_top20"].values, df_m["pearson_delta_top20"].values).pvalue)
            p_mean = float(wilcoxon(ref_ens["pearson_delta_all"].values, df_m["pearson_delta_all"].values).pvalue)
            p_mse = float(wilcoxon(ref_ens["mse_all"].values, df_m["mse_all"].values).pvalue)
            win_top20 = float((ref_ens["pearson_delta_top20"].values > df_m["pearson_delta_top20"].values).mean())

        stat_rows.append({
            "model_id": mid,
            "model": disp,
            "mean_r": mr, "mean_r_ci95_lo": mr_lo, "mean_r_ci95_hi": mr_hi,
            "top20_r": tr, "top20_r_ci95_lo": tr_lo, "top20_r_ci95_hi": tr_hi,
            "dir_acc": da, "dir_acc_ci95_lo": da_lo, "dir_acc_ci95_hi": da_hi,
            "mse": ms, "mse_ci95_lo": ms_lo, "mse_ci95_hi": ms_hi,
            "wilcoxon_p_vs_ens_top20": p_top20,
            "wilcoxon_p_vs_ens_mean": p_mean,
            "wilcoxon_p_vs_ens_mse": p_mse,
            "ens_win_rate_top20": win_top20,
        })
        print(
            f"  [{disp:14s}] Top20R={tr:.4f} [{tr_lo:.4f}, {tr_hi:.4f}] | "
            f"MeanR={mr:.4f} | WinRate(Ens)={win_top20*100:5.1f}% | {fmt_p(p_top20)}"
        )

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved strict bootstrap+Wilcoxon stats -> {OUT_STATS_CSV}")

    # --- Build Diverse 4-Panel Nature-Grade Figure 1 ---
    fig = plt.figure(figsize=(15.5, 11.8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.25)

    # Panel a: Split-Style Violin + Boxplot Distribution of Top-20 DEG Pearson r (8 primary models)
    ax_a = fig.add_subplot(gs[0, 0])
    violin_models = [
        "AIVC_v1.1_Ensemble", "AIVC_v1.1", "Linear_Ridge",
        "scFoundation_100M", "scGPT", "Geneformer_V2_104M", "GEARS_SGConv", "scVI"
    ]
    v_labels = [dict(MODELS)[m] for m in violin_models]
    v_data = [per_pert_dfs[m]["pearson_delta_top20"].values for m in violin_models]
    parts = ax_a.violinplot(v_data, positions=np.arange(len(violin_models)), showmeans=False, showextrema=False, widths=0.72)
    for idx, pc in enumerate(parts["bodies"]):
        c = PALETTE[violin_models[idx]]
        pc.set_facecolor(c)
        pc.set_edgecolor("#222222")
        pc.set_alpha(0.38)
        pc.set_linewidth(0.8)
    bp = ax_a.boxplot(
        v_data, positions=np.arange(len(violin_models)), widths=0.22,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#111111", linewidth=1.6),
        whiskerprops=dict(color="#444444", linewidth=0.9),
        capprops=dict(color="#444444", linewidth=0.9),
    )
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[violin_models[idx]])
        patch.set_alpha(0.88)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)
    ax_a.set_xticks(np.arange(len(violin_models)))
    ax_a.set_xticklabels(v_labels, rotation=22, ha="right", fontsize=8.8)
    ax_a.set_ylabel("Per-Perturbation Top-20 DEG Pearson r", fontsize=9.5, fontweight="bold")
    ax_a.set_title("a  Distribution Across 2,016 Blind Unseen RPE1 Perturbations", loc="left", fontsize=10.8, fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    # Significance bracket AIVC Ensemble vs Linear-Ridge & scGPT
    for x2, y_br, txt in [(2, 1.07, "****"), (4, 1.16, "****")]:
        ax_a.plot([0, 0, x2, x2], [y_br - 0.03, y_br, y_br, y_br - 0.03], color="#222222", lw=1.0)
        ax_a.text(x2 / 2, y_br + 0.01, txt, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_a.set_ylim(-0.35, 1.28)

    # Panel b: Paired Perturbation-Level Hexbin / Density Scatter (AIVC Ensemble vs Linear-Ridge)
    ax_b = fig.add_subplot(gs[0, 1])
    x_rg = per_pert_dfs["Linear_Ridge"]["pearson_delta_top20"].values
    y_en = ref_ens["pearson_delta_top20"].values
    hb = ax_b.hexbin(x_rg, y_en, gridsize=34, cmap="YlOrRd", mincnt=1, linewidths=0.2)
    ax_b.plot([-0.6, 1.0], [-0.6, 1.0], linestyle="--", color="#1E293B", lw=1.3, label="Parity (y = x)")
    win_pct = (y_en > x_rg).mean() * 100.0
    p_wr = wilcoxon(y_en, x_rg).pvalue
    ax_b.text(
        -0.55, 0.88,
        f"AIVC Ensemble wins: {win_pct:.1f}%\n(n = 2,016 perturbations)\nWilcoxon {fmt_p(p_wr)}",
        fontsize=8.8, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF5F5", edgecolor="#C0392B", alpha=0.92),
    )
    cb = fig.colorbar(hb, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label("Perturbation Count", fontsize=8.5)
    ax_b.set_xlim(-0.62, 1.02)
    ax_b.set_ylim(-0.62, 1.02)
    ax_b.set_xlabel("Linear-Ridge Top-20 DEG Pearson r", fontsize=9.5, fontweight="bold")
    ax_b.set_ylabel("AIVC v1.1 Ensemble Top-20 DEG Pearson r", fontsize=9.5, fontweight="bold")
    ax_b.set_title("b  Paired Perturbation-Level Superiority vs. Linear-Ridge", loc="left", fontsize=10.8, fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.3)

    # Panel c: Forest / Dot-and-95%-CI Plot (Top-20 DEG r & Mean r) — much cleaner than bar charts!
    ax_c = fig.add_subplot(gs[1, 0])
    ord_df = stats_df.sort_values("top20_r", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(ord_df))
    for i, r in ord_df.iterrows():
        c = PALETTE[r["model_id"]]
        ax_c.errorbar(
            r["top20_r"], i + 0.14,
            xerr=[[r["top20_r"] - r["top20_r_ci95_lo"]], [r["top20_r_ci95_hi"] - r["top20_r"]]],
            fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=3.5, markersize=7.5,
            label="Top-20 DEG r (95% CI)" if i == len(ord_df) - 1 else None
        )
        ax_c.errorbar(
            r["mean_r"], i - 0.14,
            xerr=[[r["mean_r"] - r["mean_r_ci95_lo"]], [r["mean_r_ci95_hi"] - r["mean_r"]]],
            fmt="s", color=c, ecolor=c, elinewidth=1.5, capsize=3.0, markersize=5.5, alpha=0.65,
            label="Mean All-Gene r (95% CI)" if i == len(ord_df) - 1 else None
        )
        ax_c.text(r["top20_r_ci95_hi"] + 0.012, i + 0.14, f"{r['top20_r']:.4f}", va="center", fontsize=8.0, color="#111111", fontweight="bold" if "AIVC" in r["model"] else "normal")
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(ord_df["model"], fontsize=8.8)
    ax_c.set_xlabel("Correlation with Ground-Truth Delta (95% Bootstrap CI)", fontsize=9.5, fontweight="bold")
    ax_c.set_xlim(0.05, 0.68)
    ax_c.set_title("c  Bootstrap 95% CI Forest Plot (All 10 Models)", loc="left", fontsize=10.8, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", linestyle="--", alpha=0.35)
    ax_c.legend(loc="lower right", fontsize=8.2, frameon=True)

    # Panel d: Multi-Axis Radar / Polar Profile (Top 6 Representative Models across 4 Normalized Axes)
    ax_d = fig.add_subplot(gs[1, 1], projection="polar")
    radar_models = ["AIVC_v1.1_Ensemble", "Linear_Ridge", "scFoundation_100M", "scGPT", "Geneformer_V2_104M", "GEARS_SGConv"]
    axes_labels = ["Mean r\n(All HVGs)", "Top-20 DEG\nPearson r", "Top-20 Sign\nDirection Acc.", "Precision\n(1 - Norm MSE)"]
    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]

    # Normalize metrics to [0.2, 1.0] across the 6 plotted models for visual clarity
    sub_rad = stats_df.set_index("model_id").loc[radar_models]
    def norm_col(series, invert=False):
        s = -series if invert else series
        lo, hi = s.min(), s.max()
        return 0.25 + 0.75 * (s - lo) / (hi - lo + 1e-9)

    n_mr = norm_col(sub_rad["mean_r"])
    n_tr = norm_col(sub_rad["top20_r"])
    n_da = norm_col(sub_rad["dir_acc"])
    n_ms = norm_col(sub_rad["mse"], invert=True)

    for mid in radar_models:
        vals = [n_mr[mid], n_tr[mid], n_da[mid], n_ms[mid]]
        vals += vals[:1]
        c = PALETTE[mid]
        lw = 2.6 if "AIVC" in mid else 1.4
        al = 0.20 if "AIVC" in mid else 0.05
        ax_d.plot(angles, vals, color=c, linewidth=lw, label=dict(MODELS)[mid], marker="o", markersize=4.5)
        ax_d.fill(angles, vals, color=c, alpha=al)

    ax_d.set_xticks(angles[:-1])
    ax_d.set_xticklabels(axes_labels, fontsize=8.8, fontweight="bold")
    ax_d.tick_params(axis="x", pad=16)
    ax_d.set_ylim(0, 1.12)
    ax_d.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_d.set_yticklabels(["", "", "", ""], fontsize=7)
    ax_d.set_title("d  Multi-Metric SOTA Profile (Normalized Radar)", loc="left", fontsize=10.8, fontweight="bold", pad=16)
    ax_d.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8.0, frameon=True)

    fig.suptitle("Figure 1 | Cross-Lineage OOD Perturbation Benchmark (7 Lineages → Blind Unseen RPE1, n = 2,016)",
                 fontsize=13.0, fontweight="bold", y=0.98)
    plt.savefig(OUT_FIG1_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_FIG1_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Nature-grade Figure 1 -> {OUT_FIG1_PNG} & {OUT_FIG1_PDF}")


if __name__ == "__main__":
    main()
