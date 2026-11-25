# -*- coding: utf-8 -*-
"""
exp45b_target_discovery_full_audit.py
Corrected, fully-reported replacement of exp45 (which reported a hand-picked subset in Tier 3).
What is evaluated (all 8 hubs x 10 TCGA cancers = 80 pre-defined pairs, NO selection after seeing results):
  - OmniPath causal-footprint score of a hub (training-free: z-scored RNA @ signed 1-hop OmniPath targets of the hub)
  - Pearson r with the hub's measured RPPA phosphosite within each cancer
  - Baseline 1: hub's own mRNA (cis) vs its phosphosite
  - Baseline 2: permutation null (500 random re-assignments of the same footprint weights to random genes)
  - Benjamini-Hochberg FDR over all 80 tests
  - Q4-vs-Q1 stratification (Mann-Whitney) for ALL 80 pairs with BH-FDR
  - Canonical-target enrichment at several cut-offs, with explicit definition of 'approved target'
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
OUT_FULL = r"D:\data\AIVC\test_results\exp45b_full_80pair_results.csv"
OUT_ENR = r"D:\data\AIVC\test_results\exp45b_enrichment_summary.csv"
OUT_PNG = r"D:\data\AIVC\test_results\nature_domain_fig6_target_discovery_3tier.png"
OUT_PDF = r"D:\data\AIVC\test_results\nature_domain_fig6_target_discovery_3tier.pdf"
N_PERM = 3000
SEED = 42

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    rng = np.random.RandomState(SEED)
    ck = np.load(CACHE_NPZ, allow_pickle=True)
    rna_genes = [str(g).upper() for g in ck["top_rna"]]
    g2i = {g: i for i, g in enumerate(rna_genes)}
    G = len(rna_genes)
    rppa_abs = [str(a) for a in ck["tcga_rppa_abs"]]
    a2i = {a: i for i, a in enumerate(rppa_abs)}
    X = ck["X_tcga"].astype(np.float32)
    Y = ck["Y_tcga_rppa"].astype(np.float32)
    C = np.array([str(c) for c in ck["C_tcga"]])
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    edges = pd.read_parquet(os.path.join(PRIORS_DIR, "directed_signaling_edges.parquet"))
    sc, tc, gc = edges.columns[0], edges.columns[1], edges.columns[2]
    edges[sc] = edges[sc].astype(str).str.upper()
    edges[tc] = edges[tc].astype(str).str.upper()

    hub_to_rppa = {"EGFR": "EGFR_pY1068", "ERBB2": "HER2_pY1248", "AKT1": "AKT_pS473", "MTOR": "MTOR_pS2448",
                   "STAT3": "STAT3_pY705", "SRC": "SRC_pY527", "YAP1": "YAP_pS127", "MET": "CMET_pY1235"}
    cancers = ["LUAD", "BRCA", "OV", "COAD", "SKCM", "HNSC", "BLCA", "STAD", "LUSC", "KIRC"]

    foot = {}
    for hub in hub_to_rppa:
        f = np.zeros(G, dtype=np.float32)
        for _, r in edges[edges[sc] == hub].iterrows():
            ti = g2i.get(r[tc], -1)
            if ti >= 0:
                f[ti] += float(r[gc])
        nrm = np.linalg.norm(f)
        foot[hub] = f / nrm if nrm > 1e-6 else f

    # permuted footprints per hub (same multiset of weights, random gene identity)
    perm_mat = {h: np.stack([f[rng.permutation(G)] for _ in range(N_PERM)], axis=1).astype(np.float32) for h, f in foot.items()}

    rows = []
    for c_name in cancers:
        cm = (C == c_name)
        for hub, rcol in hub_to_rppa.items():
            if rcol not in a2i:
                continue
            y = Y[cm, a2i[rcol]]
            xc = Xz[cm]
            valid = ~np.isnan(y)
            y, xc = y[valid], xc[valid]
            if len(y) < 30 or np.std(y) < 1e-6:
                continue
            s = xc @ foot[hub]
            if np.std(s) < 1e-6:
                continue
            r_obs, p_obs = pearsonr(s, y)
            # permutation null
            sn = xc @ perm_mat[hub]
            sn = (sn - sn.mean(0)) / (sn.std(0) + 1e-9)
            yz = (y - y.mean()) / (y.std() + 1e-9)
            r_null = (sn * yz[:, None]).mean(0)
            p_emp = (1 + np.sum(np.abs(r_null) >= abs(r_obs))) / (N_PERM + 1)
            # cis baseline
            r_cis = np.nan
            if hub in g2i:
                x_cis = xc[:, g2i[hub]]
                if np.std(x_cis) > 1e-6:
                    r_cis = pearsonr(x_cis, y)[0]
            # Q4 vs Q1
            q25, q75 = np.quantile(s, [0.25, 0.75])
            hi_g, lo_g = y[s >= q75], y[s <= q25]
            p_mw = mannwhitneyu(hi_g, lo_g, alternative="greater")[1]
            rows.append({"Cancer": c_name, "Hub": hub, "Phosphosite": rcol, "N": int(len(y)),
                         "r_footprint": float(r_obs), "p_pearson": float(p_obs),
                         "p_perm_null": float(p_emp), "null_r_mean": float(r_null.mean()), "null_r_sd": float(r_null.std()),
                         "r_cis_mRNA": float(r_cis), "Q4_minus_Q1": float(hi_g.mean() - lo_g.mean()), "p_MW_Q4gtQ1": float(p_mw)})
    df = pd.DataFrame(rows)
    df["FDR_perm"] = bh_fdr(df["p_perm_null"].values)
    df["FDR_pearson"] = bh_fdr(df["p_pearson"].values)
    df["FDR_MW"] = bh_fdr(df["p_MW_Q4gtQ1"].values)
    df["sig_positive"] = (df["r_footprint"] > 0) & (df["FDR_perm"] < 0.05)
    df.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")

    print(f"Total pairs evaluated: {len(df)}")
    print(f"  r_footprint > 0: {(df.r_footprint > 0).sum()}  |  positive & FDR_perm<0.05: {df.sig_positive.sum()}  |  negative & FDR_perm<0.05: {((df.r_footprint < 0) & (df.FDR_perm < 0.05)).sum()}")
    print(f"  Q4>Q1 significant (FDR_MW<0.05): {(df.FDR_MW < 0.05).sum()} / {len(df)}")
    print("  Per-hub: n positive-significant / n cancers")
    print(df.groupby("Hub").agg(n=("r_footprint", "size"), pos_sig=("sig_positive", "sum"), median_r=("r_footprint", "median"), median_null_sd=("null_r_sd", "median"), min_p_perm=("p_perm_null", "min"), median_cis=("r_cis_mRNA", "median")).round(4).to_string())
    print("  Pairs with p_perm<0.05 (uncorrected):", int((df.p_perm_null < 0.05).sum()), "| expected by chance at 80 tests:", 80 * 0.05)
    d2 = df.dropna(subset=["r_cis_mRNA"])
    print(f"  Footprint vs cis-mRNA on {len(d2)} pairs: footprint better in {(d2.r_footprint > d2.r_cis_mRNA).sum()}, median r footprint={d2.r_footprint.median():.3f} vs cis={d2.r_cis_mRNA.median():.3f}")
    print("\n  LUAD rows (case-study cancer):")
    print(df[df.Cancer == "LUAD"][["Hub", "Phosphosite", "N", "r_footprint", "p_perm_null", "FDR_perm", "r_cis_mRNA", "Q4_minus_Q1", "FDR_MW"]].round(4).to_string(index=False))

    # ---------------- Tier 2: enrichment ----------------
    dm = pd.read_parquet(TAHOE_DRUG)
    app = dm[dm["human-approved"].astype(str).str.lower() == "yes"]
    approved_targets = set()
    for tg in app["targets"].dropna():
        for t in str(tg).replace(";", ",").split(","):
            t = t.strip().upper()
            if t and t != "NONE":
                approved_targets.add(t)
    w = pd.read_csv(EXP28_W)
    w = w[w["regulator_type"] == "RNA_Gene"].copy()
    w["g"] = w["name"].astype(str).str.upper()
    w = w.sort_values("learned_weight", ascending=False).reset_index(drop=True)
    w["rank"] = np.arange(1, len(w) + 1)
    w["approved"] = w["g"].isin(approved_targets)
    enr = []
    for k in [100, 200, 300, 500]:
        a = int(w.iloc[:k]["approved"].sum()); b = k - a
        c = int(w.iloc[k:]["approved"].sum()); d = len(w) - k - c
        orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        enr.append({"TopK": k, "approved_in_topK": a, "approved_in_rest": c, "background_genes": len(w), "OddsRatio": orr, "Fisher_p": p})
    enr = pd.DataFrame(enr)
    enr.to_csv(OUT_ENR, index=False, encoding="utf-8-sig")
    print(f"\nApproved-drug target set: {len(approved_targets)} genes from {len(app)} approved Tahoe drugs; "
          f"{int(w.approved.sum())}/{len(w)} of the 2,893 RNA regulators are in it")
    print(enr.round(4).to_string(index=False))
    for g in ["EGFR", "ERBB2", "KIT", "EREG", "CDKN2A", "CDKN2B", "BRAF", "MAP2K1", "AKT1"]:
        rr = w[w.g == g]
        print(f"  {g}: " + (f"rank {int(rr['rank'].iloc[0])}/{len(w)}, weight {rr['learned_weight'].iloc[0]:.3f}, approved={bool(rr['approved'].iloc[0])}" if len(rr) else "not among 2,893 RNA regulators"))

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), dpi=300)
    fig.subplots_adjust(wspace=0.42)
    piv = df.pivot(index="Hub", columns="Cancer", values="r_footprint").loc[list(hub_to_rppa), cancers]
    sig = df.pivot(index="Hub", columns="Cancer", values="sig_positive").loc[list(hub_to_rppa), cancers]
    im = axes[0].imshow(piv.values, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    axes[0].set_xticks(range(len(cancers))); axes[0].set_xticklabels(cancers, rotation=38, ha="right", fontsize=8, fontweight="bold")
    axes[0].set_yticks(range(len(piv.index))); axes[0].set_yticklabels([f"{h}\n({hub_to_rppa[h]})" for h in piv.index], fontsize=7.4, fontweight="bold")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            star = "*" if bool(sig.values[i, j]) else ""
            axes[0].text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=6.6, fontweight="bold", color="white" if abs(v) > 0.28 else "#111")
    axes[0].set_title("a  All 80 hub-cancer pairs (footprint vs measured RPPA r)\n* = positive and FDR<0.05 vs permutation null", loc="left", fontsize=8.8, fontweight="bold")
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="Pearson r")

    top_app = w[w.approved].head(10)
    show = pd.concat([top_app, w[w.g.isin(["EGFR", "ERBB2"])]]).drop_duplicates("g").sort_values("learned_weight")
    yp = np.arange(len(show))
    colb = ["#2B5B84" if a else "#16A085" for a in show["approved"]]
    axes[1].hlines(yp, 0, show["learned_weight"], color=colb, lw=2.4)
    axes[1].scatter(show["learned_weight"], yp, s=80, c=colb, edgecolors="#111", lw=0.8, zorder=4)
    axes[1].set_yticks(yp); axes[1].set_yticklabels([f"{g} (#{r})" for g, r in zip(show["g"], show["rank"])], fontsize=8, fontweight="bold")
    r300 = enr[enr.TopK == 300].iloc[0]
    axes[1].set_xlabel("L1-sparse regulator weight (rank among 2,893)", fontsize=8.6, fontweight="bold")
    axes[1].set_title(f"b  Targets of approved drugs in Tahoe-100M\nTop-300 enrichment OR={r300.OddsRatio:.2f}, p={r300.Fisher_p:.1e}", loc="left", fontsize=8.8, fontweight="bold")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False); axes[1].grid(True, axis="x", ls=":", alpha=0.4)

    d2 = df.dropna(subset=["r_cis_mRNA"])
    colc = np.where(d2["sig_positive"], "#C0392B", "#95A5A6")
    axes[2].scatter(d2["r_cis_mRNA"], d2["r_footprint"], s=38, c=colc, edgecolors="#111", lw=0.5)
    lim = [-0.55, 0.55]
    axes[2].plot(lim, lim, "--", color="#2C3E50", lw=1.0); axes[2].axhline(0, color="#BDC3C7", lw=0.8); axes[2].axvline(0, color="#BDC3C7", lw=0.8)
    axes[2].set_xlim(lim); axes[2].set_ylim(lim)
    axes[2].set_xlabel("Baseline: hub's own mRNA vs phosphosite (r)", fontsize=8.6, fontweight="bold")
    axes[2].set_ylabel("OmniPath footprint score vs phosphosite (r)", fontsize=8.6, fontweight="bold")
    axes[2].set_title("c  Footprint vs cis-mRNA baseline (each dot = one pair;\nred = FDR<0.05 positive)", loc="left", fontsize=8.8, fontweight="bold")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)
    fig.suptitle("Target-Discovery Audit: all pre-defined hub-cancer pairs, cis-mRNA baseline and permutation null", fontsize=11.5, fontweight="bold", y=1.03)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight"); fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT_FULL, OUT_ENR, OUT_PNG)


if __name__ == "__main__":
    main()
