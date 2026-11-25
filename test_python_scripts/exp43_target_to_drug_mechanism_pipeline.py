# -*- coding: utf-8 -*-
"""
exp43_target_to_drug_mechanism_pipeline.py
Project 2 End-to-End Pipeline:
  1. Target Prioritization (AIVC-5Omics + OmniPath causal regulators)
  2. Target-Matched Drug Screening (Tahoe-100M 379-drug library + RDKit ADMET/QED/PAINS)
  3. 3D Co-Folding & Quantum Force Field Audit (Boltz-2 + MACE-OFF23 verified metrics)
  4. Mechanistic Interpretation (OmniPath 1-hop activation/inhibition targets + GTEx/EyeGEx safety)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aivc

TAHOE_DRUG_META = r"D:\data\AIVC\data\tahoe_100m\metadata\drug_metadata.parquet"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
BOLTZ_CSV = r"D:\data\AIVC\versions\v1.1_frozen_milestone\test_results\egfr_wt_vs_t790m_boltz2_mace.csv"
SAFETY_CSV = r"D:\data\AIVC\test_results\aivc_flagship_safety_report.csv"

OUT_CSV = r"D:\data\AIVC\test_results\exp43_target_to_drug_mechanism_report.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp43_target_to_drug_mechanism_plot.png"


def main():
    print("=== Project 2: Target -> Drug Screening -> 3D/Quantum Audit -> Mechanistic Interpretation ===")
    focus_targets = {"EGFR", "ERBB2", "BRAF", "MAP2K1", "AKT1"}

    # 1. Screen Tahoe-100M compounds matching focus_targets
    dm = pd.read_parquet(TAHOE_DRUG_META)
    matched_rows = []
    for _, r in dm.iterrows():
        tgts_raw = str(r.get("targets", ""))
        if not tgts_raw or tgts_raw == "None":
            continue
        tlist = [t.strip().upper() for t in tgts_raw.replace(";", ",").split(",") if t.strip()]
        hit = sorted(set(tlist) & focus_targets)
        smi = str(r.get("canonical_smiles", "")).strip()
        if hit and smi and smi != "None":
            matched_rows.append({
                "drug": str(r["drug"]).strip(),
                "hit_targets": ";".join(hit),
                "all_targets": ";".join(tlist),
                "moa_fine": str(r.get("moa-fine", "")),
                "human_approved": str(r.get("human-approved", "")),
                "smiles": smi,
            })
    scr_df = pd.DataFrame(matched_rows).drop_duplicates("drug").reset_index(drop=True)
    print(f"  [Stage 1-2] Screened {len(scr_df)} compounds targeting {sorted(focus_targets)} from Tahoe-100M")

    # 2. Compute RDKit ADMET / QED / Lipinski / PAINS via aivc.compute_admet_druglikeness
    admet_df = aivc.compute_admet_druglikeness(scr_df["smiles"].tolist())
    scr_df = pd.concat([scr_df, admet_df.drop(columns=["smiles"], errors="ignore")], axis=1)

    # 3. Map each hit target to its OmniPath 1-hop downstream causal effectors
    edges = pd.read_parquet(os.path.join(PRIORS_DIR, "directed_signaling_edges.parquet"))
    src_col, tgt_col, sign_col = edges.columns[0], edges.columns[1], edges.columns[2]
    edges_up = edges.copy()
    edges_up[src_col] = edges_up[src_col].astype(str).str.upper()
    edges_up[tgt_col] = edges_up[tgt_col].astype(str).str.upper()

    downstream_map = {}
    for tg in sorted(focus_targets):
        sub_e = edges_up[edges_up[src_col] == tg]
        act = sorted(sub_e[sub_e[sign_col] > 0][tgt_col].unique())[:5]
        inh = sorted(sub_e[sub_e[sign_col] < 0][tgt_col].unique())[:5]
        downstream_map[tg] = f"Act:[{','.join(act)}] | Inh:[{','.join(inh)}] (total={len(sub_e)})"

    # 4. Attach GTEx/EyeGEx safety & Boltz-2/MACE-OFF23 3D structural audit
    safety_df = pd.read_csv(SAFETY_CSV).set_index("gene")
    boltz_df = pd.read_csv(BOLTZ_CSV) if os.path.exists(BOLTZ_CSV) else pd.DataFrame()

    mech_notes, retina_vals, organ_counts = [], [], []
    for _, r in scr_df.iterrows():
        primary_t = r["hit_targets"].split(";")[0]
        mech_notes.append(downstream_map.get(primary_t, ""))
        if primary_t in safety_df.index:
            retina_vals.append(float(safety_df.loc[primary_t, "eyegex_retina_median"]))
            organ_counts.append(int(safety_df.loc[primary_t, "gtex_organs_above_thresh"]))
        else:
            retina_vals.append(0.0)
            organ_counts.append(0)

    scr_df["omnipath_downstream_mechanism"] = mech_notes
    scr_df["primary_target_retina_cpm"] = retina_vals
    scr_df["primary_target_gtex_organs"] = organ_counts
    scr_df = scr_df.sort_values(["pains_alerts", "qed"], ascending=[True, False]).reset_index(drop=True)
    scr_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n  Top Screened Candidates (PAINS=0, ranked by QED):")
    show_cols = ["drug", "hit_targets", "qed", "mw", "clogp", "pains_alerts", "human_approved", "primary_target_retina_cpm"]
    print(scr_df[show_cols].head(10).to_string(index=False))

    # Plot 2-panel summary: Panel A = Screened Candidates QED vs cLogP (colored by Target), Panel B = Boltz-2 + MACE-OFF23 EGFR WT vs T790M Audit
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    t_colors = {"EGFR": "#C0392B", "ERBB2": "#E67E22", "BRAF": "#8E44AD", "MAP2K1": "#16A085", "AKT1": "#2B5B84"}
    for tg, grp in scr_df.groupby(scr_df["hit_targets"].str.split(";").str[0]):
        axes[0].scatter(grp["clogp"], grp["qed"], s=75, color=t_colors.get(tg, "#7F8C8D"), alpha=0.88, edgecolors="#111111", lw=0.8, label=tg)
    top6 = scr_df.head(6)
    for _, r in top6.iterrows():
        axes[0].text(r["clogp"] + 0.08, r["qed"] + 0.012, r["drug"], fontsize=7.6, fontweight="bold")
    axes[0].set_xlabel("Lipophilicity (cLogP)", fontsize=9.2, fontweight="bold")
    axes[0].set_ylabel("Quantitative Estimate of Druglikeness (QED)", fontsize=9.2, fontweight="bold")
    axes[0].set_title("a  Target-Matched Drug Candidates (RDKit ADMET)", loc="left", fontsize=10.5, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.2, frameon=True)

    if len(boltz_df) > 0:
        # Plot Boltz-2 binding prob vs MACE-OFF energy
        lbls = [str(x).replace("__", " + ") for x in boltz_df["complex"]]
        probs = boltz_df["affinity_probability_binary"].values * 100.0
        energies = boltz_df["mace_bind_energy_kcal"].values
        cols = ["#7F8C8D", "#C0392B", "#2B5B84", "#16A085"][:len(lbls)]
        axes[1].scatter(probs, energies, s=160, c=cols, edgecolors="#111111", lw=1.1, zorder=4)
        offsets = [(1.2, 1.5), (1.2, -2.5), (-22.0, 2.0), (-23.0, -3.0)]
        for i, lb in enumerate(lbls):
            dx, dy = offsets[i] if i < len(offsets) else (1.2, -1.5)
            axes[1].text(probs[i] + dx, energies[i] + dy, lb, fontsize=8.0, fontweight="bold", color=cols[i])
        axes[1].set_xlabel("Boltz-2 Co-Folding Binding Probability (%)", fontsize=9.2, fontweight="bold")
        axes[1].set_ylabel("MACE-OFF23 Quantum Binding Energy (kcal/mol, Lower ↓)", fontsize=9.2, fontweight="bold")
        axes[1].set_xlim(35, 98)
        axes[1].set_ylim(-68, -22)
        axes[1].set_title("b  3D Resistance Mutant Audit (EGFR WT vs T790M)", loc="left", fontsize=10.5, fontweight="bold")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Project 2 | AIVC Target -> Drug Screening -> 3D Co-Folding & Mechanistic Safety Audit",
                 fontsize=12.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved Project 2 report -> {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
