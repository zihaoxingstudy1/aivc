# -*- coding: utf-8 -*-
"""make_ablation_forest_fig.py : paired-bootstrap ablation forest plots (exp49 RPE1 + exp50 combos)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = r"D:\data\AIVC\test_results"
a = pd.read_csv(os.path.join(R, "exp49_rpe1_paired_bootstrap.csv"))
b = pd.read_csv(os.path.join(R, "exp50_combo_paired_bootstrap.csv"))
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

label_a = {
    "no_prior": "Remove OmniPath prior (signed)", "no_5omics_state": "Remove 5-omics baseline state",
    "no_esm2": "Remove ESM-2 embedding", "no_coexp": "Remove co-expression embedding",
    "no_teach_all": "Remove ESM-2 + co-expression", "no_ctrl": "Remove control-profile input",
    "all_inputs_zero": "All inputs zeroed (anchor only)", "mse_only": "Drop cosine loss (post hoc; not adopted)",
    "residual_ridge(alpha=100)": "Linear residual on identical features (alpha=100)", "ridge_only": "Linear-Ridge anchor only",
    "no_tstat_all": "Remove all 5 target statistics",
}
order_a = ["no_prior", "no_5omics_state", "no_tstat_all", "no_esm2", "no_coexp", "no_teach_all", "no_ctrl", "all_inputs_zero",
           "residual_ridge(alpha=100)", "ridge_only", "mse_only"]
a = a.set_index("Arm_vs_full_signed").loc[[o for o in order_a if o in set(a["Arm_vs_full_signed"])]].reset_index()

label_b = {"no_xtalk": "Remove OmniPath cross-talk term", "xtalk_unsigned": "Cross-talk with sign removed",
           "no_dom": "Remove dominant-effect envelope", "no_esm": "Remove ESM-2 pair features",
           "no_zfeat": "Remove single-gene program features", "no_feat_input": "Head receives zeros",
           "additive_gain_only": "Gain-scaled additive only"}
order_b = ["no_xtalk", "xtalk_unsigned", "no_dom", "no_esm", "no_zfeat", "no_feat_input", "additive_gain_only"]
b = b.set_index("Arm_vs_full").loc[order_b].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), dpi=300)
fig.subplots_adjust(wspace=0.62)


def forest(ax, df, namecol, dcol, labels, xlabel, title, better_negative):
    y = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        d, lo, hi = r[dcol], r["CI95_lo"], r["CI95_hi"]
        sig = (lo > 0) or (hi < 0)
        hurts = (d < 0) if better_negative else (d > 0)
        col = ("#C0392B" if hurts else "#2B5B84") if sig else "#7F8C8D"
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.4)
        ax.scatter([d], [yi], s=70, color=col, edgecolor="#111", zorder=4)
    ax.axvline(0, color="#2C3E50", lw=1.1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(n, n) for n in df[namecol]], fontsize=8.2, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=9.6, fontweight="bold")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


forest(axes[0], a, "Arm_vs_full_signed", "Delta_Top20R_(arm-full)", label_a,
       "Change in Top-20 DEG r vs full AIVC (paired bootstrap 95% CI)",
       "a  RPE1 (n = 2,016): effect of each change", True)
forest(axes[1], b, "Arm_vs_full", "Delta_MSE(arm-full)", label_b,
       "Change in MSE vs full AIVC-Combo (95% CI)",
       "b  Norman2019 combinations (n = 27)", False)
fig.suptitle("Component attribution: what drives AIVC's gains and what does not", fontsize=11.5, fontweight="bold", y=1.02)
fig.savefig(os.path.join(R, "nature_fig_ablation_forest.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(R, "nature_fig_ablation_forest.pdf"), bbox_inches="tight")
print("saved")
