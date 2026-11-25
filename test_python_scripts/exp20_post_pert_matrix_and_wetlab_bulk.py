import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex
from post_perturbation_exporter import export_post_perturbation_h5ad, export_post_perturbation_bulk
from bio_pathway_viz import run_go_kegg_enrichment

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_BULK_DIR = r"D:\data\AIVC\data\wetlab_bulk_validation"
OUT_CSV = r"D:\data\AIVC\test_results\exp20_wetlab_bulk_consistency.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp20_wetlab_bulk_consistency.png"
OUT_H5AD = r"D:\data\AIVC\test_results\exp20_virtual_panobinostat_post.h5ad"
OUT_DEG_CSV = r"D:\data\AIVC\test_results\exp20_virtual_panobinostat_degs.csv"
OUT_ENRICH_CSV = r"D:\data\AIVC\test_results\exp20_panobinostat_kegg_go.csv"
OUT_ENRICH_PNG = r"D:\data\AIVC\test_results\exp20_panobinostat_enrichment_bubble.png"


def counts_to_log2_cpm(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert raw/norm counts (genes x samples) to log2(CPM + 1) (samples x genes)."""
    vals = df_counts.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    col_sums = np.maximum(vals.sum(axis=0, keepdims=True), 1.0)
    cpm = vals / col_sums * 1e6
    log2_cpm = np.log2(cpm + 1.0).astype(np.float32)
    return pd.DataFrame(log2_cpm.T, index=df_counts.columns, columns=df_counts.index)


class BioMolKGPertNet(nn.Module):
    """Cell baseline + Drug (Morgan + ChemBERTa + Target ESM-2) + Dose -> Delta Expr."""
    def __init__(self, n_genes: int, n_drug: int = 1408, hid: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(n_drug, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
        )
        self.dose_enc = nn.Linear(1, 32)
        self.dec = nn.Sequential(
            nn.Linear(hid * 2 + 32, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, n_genes),
        )

    def forward(self, x_ctrl, x_drug, x_dose):
        hc = self.cell_enc(x_ctrl)
        hd = self.drug_enc(x_drug)
        hdose = torch.relu(self.dose_enc(x_dose))
        return self.dec(torch.cat([hc, hd, hdose], dim=-1))


def eval_pred_vs_real_bulk(
    name: str,
    ctrl_mat: pd.DataFrame,
    real_treated_mat: pd.DataFrame,
    pred_delta_vec: np.ndarray,
    pred_genes: list,
    out_prefix: str,
) -> dict:
    """Export predicted post-perturbation Bulk matrix and evaluate consistency vs real wet-lab treated matrix."""
    post_path = os.path.join(r"D:\data\AIVC\test_results", f"exp20_{out_prefix}_pred_post.parquet")
    deg_path = os.path.join(r"D:\data\AIVC\test_results", f"exp20_{out_prefix}_pred_degs.csv")
    pred_post_df, _ = export_post_perturbation_bulk(
        ctrl_mat, pred_delta_vec, pred_genes, name, out_post_path=post_path, out_deg_csv_path=deg_path
    )

    # Align shared genes between model pred_genes and Bulk matrix
    bulk_upper = {str(c).strip().upper(): c for c in ctrl_mat.columns}
    shared = [(j, g, bulk_upper[g.upper()]) for j, g in enumerate(pred_genes) if g.upper() in bulk_upper]
    idx_p = [s[0] for s in shared]
    cols_b = [s[2] for s in shared]

    real_delta = (real_treated_mat[cols_b].mean(axis=0).values - ctrl_mat[cols_b].mean(axis=0).values).astype(np.float32)
    pred_delta = pred_delta_vec[idx_p].astype(np.float32)

    r_all, _ = pearsonr(pred_delta, real_delta)
    rho_all, _ = spearmanr(pred_delta, real_delta)
    dir_all = float(np.mean(np.sign(pred_delta) == np.sign(real_delta)))

    # Top 50 & Top 100 real DEGs
    top50_idx = np.argsort(np.abs(real_delta))[-50:]
    top100_idx = np.argsort(np.abs(real_delta))[-100:]
    r_top50, _ = pearsonr(pred_delta[top50_idx], real_delta[top50_idx])
    dir_top50 = float(np.mean(np.sign(pred_delta[top50_idx]) == np.sign(real_delta[top50_idx])))
    r_top100, _ = pearsonr(pred_delta[top100_idx], real_delta[top100_idx])
    dir_top100 = float(np.mean(np.sign(pred_delta[top100_idx]) == np.sign(real_delta[top100_idx])))

    # Also compare predicted post-treatment mean profile vs real post-treatment mean profile
    pred_post_mean = pred_post_df[cols_b].mean(axis=0).values
    real_post_mean = real_treated_mat[cols_b].mean(axis=0).values
    r_post_prof, _ = pearsonr(pred_post_mean, real_post_mean)

    print(
        f"  [{name}] Shared={len(shared)} | PostProfile r={r_post_prof:.4f} | "
        f"Delta All r={r_all:.4f} (rho={rho_all:.4f}) | "
        f"Top50 DEG r={r_top50:.4f}, DirAcc={dir_top50*100:.1f}% | Top100 DirAcc={dir_top100*100:.1f}%"
    )
    return {
        "benchmark": name,
        "n_shared_genes": len(shared),
        "post_profile_pearson": r_post_prof,
        "delta_all_pearson": r_all,
        "delta_all_spearman": rho_all,
        "delta_all_dir_acc": dir_all,
        "top50_deg_pearson": r_top50,
        "top50_deg_dir_acc": dir_top50,
        "top100_deg_pearson": r_top100,
        "top100_deg_dir_acc": dir_top100,
    }


def main():
    print(f"=== Exp20: Post-Perturbation Matrix Exporter & Wet-Lab GEO Bulk Validation ({DEVICE}) ===")
    ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")

    # 1. Load Sciplex3_24h_hvg2000 & build condition-level pseudobulk pairs (Ctrl -> Drug)
    print("Loading Sciplex3_24h_hvg2000.h5ad...")
    s3 = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad")
    sciplex_genes = [str(g) for g in s3.var_names]
    X_s3 = s3.X.toarray().astype(np.float32) if sp.issparse(s3.X) else np.asarray(s3.X, dtype=np.float32)

    m_npz = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=True)
    drug_feats = {
        str(d): np.concatenate([m_npz["morgan"][i], m_npz["chemberta"][i]]).astype(np.float32)
        for i, d in enumerate(m_npz["drugs"])
    }

    # Compute vehicle baseline per cell_line
    obs = s3.obs.copy()
    obs["pert_str"] = obs["perturbation"].astype(str)
    ctrl_mask = obs["pert_str"].str.lower().isin(["vehicle", "dmso", "control"])
    ctrl_by_cl = {}
    for cl in obs["cell_line"].unique():
        m = ctrl_mask & (obs["cell_line"] == cl)
        ctrl_by_cl[str(cl)] = X_s3[m.values].mean(axis=0)

    # Build condition pseudobulk deltas
    cond_rows = []
    grouped = obs[~ctrl_mask].groupby(["cell_line", "pert_str", "dose_value"], observed=True).indices
    for (cl, pert, dose), idxs in grouped.items():
        if len(idxs) < 5 or pert not in drug_feats:
            continue
        x_c = ctrl_by_cl[str(cl)]
        delta = X_s3[idxs].mean(axis=0) - x_c
        log_dose = float(np.log10(max(float(dose), 1.0)) / 4.0)
        cond_rows.append((str(cl), pert, float(dose), x_c, drug_feats[pert].astype(np.float32), np.array([log_dose], dtype=np.float32), delta))

    print(f"Built {len(cond_rows)} Sciplex3 condition pairs across {len(ctrl_by_cl)} cell lines.")

    def train_and_predict_panobinostat(holdout_panobinostat: bool, target_ctrl_vec: np.ndarray, dose_nm: float = 100.0) -> np.ndarray:
        tr = [r for r in cond_rows if not (holdout_panobinostat and "Panobinostat" in r[1])]
        Xc = torch.tensor(np.stack([r[3] for r in tr]), device=DEVICE)
        Xd = torch.tensor(np.stack([r[4] for r in tr]), device=DEVICE)
        Xdose = torch.tensor(np.stack([r[5] for r in tr]), device=DEVICE)
        Yd = torch.tensor(np.stack([r[6] for r in tr]), device=DEVICE)

        net = BioMolKGPertNet(len(sciplex_genes), n_drug=1408).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
        mse_fn = nn.MSELoss()
        cos_fn = nn.CosineSimilarity(dim=-1)
        net.train()
        for _ in range(100):
            perm = torch.randperm(len(Xc), device=DEVICE)
            for i in range(0, len(Xc), 256):
                b = perm[i:i+256]
                opt.zero_grad()
                pred = net(Xc[b], Xd[b], Xdose[b])
                loss = mse_fn(pred, Yd[b]) + 0.25 * (1.0 - cos_fn(pred, Yd[b])).mean()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            p_name = "Panobinostat (LBH589)"
            log_d = float(np.log10(max(dose_nm, 1.0)) / 4.0)
            xd_p = torch.tensor(drug_feats[p_name].astype(np.float32), device=DEVICE).unsqueeze(0)
            xdose_p = torch.tensor([[log_d]], dtype=torch.float32, device=DEVICE)
            xc_p = torch.tensor(target_ctrl_vec.astype(np.float32), device=DEVICE).unsqueeze(0)
            return net(xc_p, xd_p, xdose_p).squeeze(0).cpu().numpy()

    # Mean single-cell baseline for model conditioning
    mean_sc_ctrl = np.mean(list(ctrl_by_cl.values()), axis=0)
    pred_pano_seen = train_and_predict_panobinostat(holdout_panobinostat=False, target_ctrl_vec=mean_sc_ctrl, dose_nm=100.0)
    pred_pano_zeroshot = train_and_predict_panobinostat(holdout_panobinostat=True, target_ctrl_vec=mean_sc_ctrl, dose_nm=100.0)

    results = []

    # 2. Wet-Lab Bulk Benchmark 1: GSE307106 (OPM2 Myeloma, DMSO vs Panobinostat 20h)
    print("\n=== Benchmark 1: GSE307106 (OPM2 Myeloma Bulk RNA-seq, DMSO vs Panobinostat 20h) ===")
    df307 = pd.read_csv(os.path.join(RAW_BULK_DIR, "GSE307106", "GSE307106_raw_counts.txt.gz"), sep="\t", index_col=0)
    df307["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df307.index]
    df307 = df307[df307["__sym__"] != ""].groupby("__sym__", sort=False).mean()
    mat307 = counts_to_log2_cpm(df307)
    ctrl307 = mat307.loc[["DMSO_1", "DMSO_2"]]
    trt307_20h = mat307.loc[["Panobinostat_20h_1", "Panobinostat_20h_2"]]
    results.append(eval_pred_vs_real_bulk("GSE307106_OPM2_Panobinostat20h_SeenDrugTransfer", ctrl307, trt307_20h, pred_pano_seen, sciplex_genes, "gse307106_seen"))
    results.append(eval_pred_vs_real_bulk("GSE307106_OPM2_Panobinostat20h_ZeroShotBlind", ctrl307, trt307_20h, pred_pano_zeroshot, sciplex_genes, "gse307106_zs"))

    # 3. Wet-Lab Bulk Benchmark 2: GSE308634 (9 Glioma Cell Lines, Vehicle vs Panobinostat 10nM)
    print("\n=== Benchmark 2: GSE308634 (Glioma Bulk RNA-seq, Vehicle vs Panobinostat 10nM) ===")
    df308 = pd.read_csv(os.path.join(RAW_BULK_DIR, "GSE308634", "GSE308634_ALL_deseq_norm_counts_annotated.csv.gz"), index_col=0)
    num_cols = [c for c in df308.columns if ("_Veh_" in c or "_10nM_" in c)]
    df308_num = df308[num_cols].copy()
    df308_num["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df308_num.index]
    df308_num = df308_num[df308_num["__sym__"] != ""].groupby("__sym__", sort=False).mean()
    mat308 = counts_to_log2_cpm(df308_num)
    veh_cols = [c for c in num_cols if "_Veh_" in c]
    trt_cols = [c for c in num_cols if "_10nM_" in c]
    results.append(eval_pred_vs_real_bulk("GSE308634_Glioma_Panobinostat10nM_SeenDrugTransfer", mat308.loc[veh_cols], mat308.loc[trt_cols], pred_pano_seen, sciplex_genes, "gse308634_seen"))
    results.append(eval_pred_vs_real_bulk("GSE308634_Glioma_Panobinostat10nM_ZeroShotBlind", mat308.loc[veh_cols], mat308.loc[trt_cols], pred_pano_zeroshot, sciplex_genes, "gse308634_zs"))

    # 4. Wet-Lab Bulk Benchmark 3: GSE343784 (HCT116 Colon Cancer, WT vs MCL1-KO) using Replogle K562 MCL1-KO transfer
    print("\n=== Benchmark 3: GSE343784 (HCT116 Bulk RNA-seq, WT vs MCL1-KO) <- Replogle K562 MCL1-KO ===")
    adata_rep = ad.read_h5ad(r"D:\data\AIVC\data\ReplogleWeissman2022_K562_gwps.h5ad", backed="r")
    rep_genes = [str(g) for g in adata_rep.var["gene_name"]] if "gene_name" in adata_rep.var.columns else [str(g) for g in adata_rep.var_names]
    mcl1_idx = np.where(adata_rep.obs["perturbation"].astype(str).values == "MCL1")[0]
    nt_idx = np.where(adata_rep.obs["perturbation"].astype(str).values == "control")[0][:1000]
    X_mcl1 = np.asarray(adata_rep.X[mcl1_idx].toarray() if sp.issparse(adata_rep.X[mcl1_idx]) else adata_rep.X[mcl1_idx], dtype=np.float32)
    X_nt = np.asarray(adata_rep.X[nt_idx].toarray() if sp.issparse(adata_rep.X[nt_idx]) else adata_rep.X[nt_idx], dtype=np.float32)
    mcl1_delta = X_mcl1.mean(axis=0) - X_nt.mean(axis=0)

    df34 = pd.read_csv(os.path.join(RAW_BULK_DIR, "GSE343784", "GSE343784_raw_counts.tsv.gz"), sep="\t")
    df34["gene_symbol"] = df34["gene_symbol"].astype(str).str.strip().str.upper()
    count_cols34 = [c for c in df34.columns if c.startswith(("wt_", "ko_"))]
    df34 = df34[~df34["gene_symbol"].isin(["", "NAN"])].groupby("gene_symbol", sort=False)[count_cols34].mean()
    mat34 = counts_to_log2_cpm(df34)
    results.append(eval_pred_vs_real_bulk(
        "GSE343784_HCT116_MCL1_KO_CrossCellTransfer",
        mat34.loc[["wt_dmso_r1", "wt_dmso_r2", "wt_dmso_r3"]],
        mat34.loc[["ko_dmso_r1", "ko_dmso_r2", "ko_dmso_r3"]],
        mcl1_delta, rep_genes, "gse343784_mcl1ko"
    ))

    # 5. Export standard single-cell AnnData (.h5ad) + run downstream GO/KEGG visualization
    print("\n=== Exporting Scanpy-Compatible Virtual Post-Perturbation .h5ad + Downstream GO/KEGG ===")
    veh_cells = s3[ctrl_mask.values][:500].copy()
    adata_post, deg_df = export_post_perturbation_h5ad(
        veh_cells, pred_pano_seen, sciplex_genes, "Panobinostat_10nM", OUT_H5AD, OUT_DEG_CSV
    )
    print(f"  Exported {OUT_H5AD} | shape={adata_post.shape} | layers={list(adata_post.layers.keys())} | pred_genes={int(adata_post.var['is_predicted_gene'].sum())}")
    run_go_kegg_enrichment(
        pred_pano_seen, sciplex_genes, "Panobinostat_10nM_VirtualPost", OUT_ENRICH_CSV, OUT_ENRICH_PNG, top_n_genes=100
    )

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved Exp20 results to {OUT_CSV}")

    # Plot summary figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    short_names = [
        "GSE307106\n(Myeloma Pano\nSeen-Drug)",
        "GSE307106\n(Myeloma Pano\nZero-Shot)",
        "GSE308634\n(Glioma Pano\nSeen-Drug)",
        "GSE308634\n(Glioma Pano\nZero-Shot)",
        "GSE343784\n(HCT116 MCL1-KO\nCross-Cell)",
    ]
    x_pos = np.arange(len(res_df))
    axes[0].bar(x_pos - 0.18, res_df["post_profile_pearson"], width=0.35, label="Post-Pert Matrix Profile r", color="#2b7bba", edgecolor="black", linewidth=0.5)
    axes[0].bar(x_pos + 0.18, res_df["top50_deg_pearson"], width=0.35, label="Top-50 Real DEG Delta r", color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(short_names, fontsize=8)
    axes[0].set_ylabel("Pearson Correlation (r)")
    axes[0].set_title("Single-Cell Model -> External Wet-Lab Bulk RNA-seq\n(Post-Perturbation Matrix & DEG Correlation)", fontsize=10, fontweight="bold")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend(fontsize=8)

    axes[1].bar(x_pos - 0.18, res_df["top50_deg_dir_acc"] * 100, width=0.35, label="Top-50 DEG Direction Acc (%)", color="#27ae60", edgecolor="black", linewidth=0.5)
    axes[1].bar(x_pos + 0.18, res_df["top100_deg_dir_acc"] * 100, width=0.35, label="Top-100 DEG Direction Acc (%)", color="#8e44ad", edgecolor="black", linewidth=0.5)
    axes[1].axhline(50, color="red", linestyle="--", linewidth=0.9, label="Random Chance (50%)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(short_names, fontsize=8)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Up/Down Direction Agreement (%)")
    axes[1].set_title("DEG Up/Down Regulation Direction Consistency\n(vs Real Wet-Lab Treated / KO Bulk RNA-seq)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved Exp20 figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
