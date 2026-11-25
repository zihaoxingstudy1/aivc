import os
import re
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gseapy as gp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex
from exp27b_local_gated_5omics import (
    TCGA_RNA_DIR, TCGA_RPPA_PATH, TCGA_MAF_DIR, CPTAC_DIR, CPTAC_MUT_DIR,
    CCLE_RPPA_PATH, CCLE_TMT_PATH, HOTSPOT_PATH, BIOGRID_PATH, CELL_RNA_PATH,
    CMP_PATH, CACHE_24, UNSEEN_CANCERS, CPTAC_MUT_PREFIX, CORE_TFS, RPPA_TO_GENE,
    row_zscore, norm_cell_name, norm_ab_name, build_graphs_and_mut_masks
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_CKPT = r"D:\data\AIVC\pretrained\aivc_5omics_pathway_gnn.pt"
OUT_CSV = r"D:\data\AIVC\test_results\exp28_sparse_importance_results.csv"
OUT_IMP_CSV = r"D:\data\AIVC\test_results\exp28_learned_regulator_weights.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp28_learned_importance_plot.png"


class SparseReg5OmicsPathwayGNN(nn.Module):
    """
    5-Omics Pathway-GNN with Learnable L1-Sparse Node Importance Gates:
      - alpha_rna (N_rna), alpha_tf (N_tf), alpha_mir (N_mir) in (0, 1)
      - Pushes noisy background genes/miRNAs toward 0 while elevating master TFs, key miRNAs, and hub genes
      - Preserves 1-hop BioGRID-restricted Mutation->Target gates
    """
    def __init__(self, n_rna: int, n_mut: int, n_mir: int, rppa_cis: list, prot_cis: list, phos_cis: list,
                 A_tf: np.ndarray, A_gp: np.ndarray, M_rppa: np.ndarray, M_prot: np.ndarray, M_phos: np.ndarray, hid: int = 192):
        super().__init__()
        self.register_buffer("A_tf", torch.tensor(A_tf, dtype=torch.float32))
        self.register_buffer("A_gp", torch.tensor(A_gp, dtype=torch.float32))
        self.register_buffer("rppa_cis", torch.tensor(rppa_cis, dtype=torch.long))
        self.register_buffer("prot_cis", torch.tensor(prot_cis, dtype=torch.long))
        self.register_buffer("phos_cis", torch.tensor(phos_cis, dtype=torch.long))
        self.register_buffer("M_rppa", torch.tensor(M_rppa, dtype=torch.float32))
        self.register_buffer("M_prot", torch.tensor(M_prot, dtype=torch.float32))
        self.register_buffer("M_phos", torch.tensor(M_phos, dtype=torch.float32))

        # Learnable node importance logits (initialized near 0 -> sigmoid=0.5)
        self.g_rna = nn.Parameter(torch.zeros(n_rna))
        self.g_tf = nn.Parameter(torch.zeros(A_tf.shape[1]))
        self.g_mir = nn.Parameter(torch.zeros(n_mir))

        self.rppa_cis_w = nn.Parameter(torch.ones(len(rppa_cis)) * 0.40)
        self.prot_cis_w = nn.Parameter(torch.ones(len(prot_cis)) * 0.55)
        self.phos_cis_w = nn.Parameter(torch.ones(len(phos_cis)) * 0.25)

        self.pw_enc = nn.Sequential(nn.Linear(A_gp.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.tf_enc = nn.Sequential(nn.Linear(A_tf.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.glob_enc = nn.Sequential(nn.Linear(n_rna, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.25))
        self.fuse_rna = nn.Sequential(nn.Linear(hid * 3, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.20))

        self.rppa_dec = nn.Linear(hid, len(rppa_cis))
        self.prot_dec = nn.Linear(hid, len(prot_cis))
        self.phos_dec = nn.Linear(hid, len(phos_cis))
        nn.init.zeros_(self.rppa_dec.weight)
        nn.init.zeros_(self.prot_dec.weight)
        nn.init.zeros_(self.phos_dec.weight)

        self.W_mut_rppa = nn.Parameter(torch.zeros(n_mut, len(rppa_cis)))
        self.W_mut_prot = nn.Parameter(torch.zeros(n_mut, len(prot_cis)))
        self.W_mut_phos = nn.Parameter(torch.zeros(n_mut, len(phos_cis)))

        self.mir_enc = nn.Sequential(nn.Linear(n_mir, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.25))
        self.mir_prot = nn.Linear(32, len(prot_cis), bias=False)
        self.mir_phos = nn.Linear(32, len(phos_cis), bias=False)
        nn.init.zeros_(self.mir_prot.weight)
        nn.init.zeros_(self.mir_phos.weight)

    def get_gates(self):
        return torch.sigmoid(self.g_rna), torch.sigmoid(self.g_tf), torch.sigmoid(self.g_mir)

    def encode_rna(self, x_rna: torch.Tensor) -> torch.Tensor:
        a_rna, a_tf, _ = self.get_gates()
        xg = x_rna * a_rna
        h_pw = self.pw_enc(xg @ self.A_gp)
        h_tf = self.tf_enc((xg @ self.A_tf) * a_tf)
        h_gl = self.glob_enc(xg)
        return self.fuse_rna(torch.cat([h_pw, h_tf, h_gl], dim=-1))

    def forward_rppa(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.rppa_cis >= 0).float()
        si = torch.clamp(self.rppa_cis, min=0)
        out = x_rna[:, si] * vm * self.rppa_cis_w + 0.45 * self.rppa_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_rppa * self.M_rppa)
        return out

    def forward_prot(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.prot_cis >= 0).float()
        si = torch.clamp(self.prot_cis, min=0)
        out = x_rna[:, si] * vm * self.prot_cis_w + 0.35 * self.prot_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_prot * self.M_prot)
        if x_mir is not None:
            _, _, a_mir = self.get_gates()
            out = out + 0.20 * self.mir_prot(self.mir_enc(x_mir * a_mir))
        return out

    def forward_phos(self, x_rna, x_mut=None, x_mir=None):
        h = self.encode_rna(x_rna)
        vm = (self.phos_cis >= 0).float()
        si = torch.clamp(self.phos_cis, min=0)
        out = x_rna[:, si] * vm * self.phos_cis_w + 0.35 * self.phos_dec(h)
        if x_mut is not None:
            out = out + x_mut @ (self.W_mut_phos * self.M_phos)
        if x_mir is not None:
            _, _, a_mir = self.get_gates()
            out = out + 0.20 * self.mir_phos(self.mir_enc(x_mir * a_mir))
        return out


def main():
    print(f"=== Exp28: Sparse L1 Importance Gates + Regulatory Weight Discovery ({DEVICE}) ===")

    hot = pd.read_csv(HOTSPOT_PATH)
    driver_genes = [str(c).split(" (")[0].strip().upper() for c in hot.columns[6:]]
    driver_set = set(driver_genes)
    drv_pos = {g: i for i, g in enumerate(driver_genes)}

    c24 = np.load(CACHE_24, allow_pickle=True)
    top_rna = [str(g) for g in c24["top_rna"]]
    top_prot = [str(g) for g in c24["top_prot"]]
    rna_pos = {g: i for i, g in enumerate(top_rna)}
    ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")

    # Load TCGA
    rppa = pd.read_csv(TCGA_RPPA_PATH, sep="\t", index_col=0)
    rppa = rppa[rppa.isna().mean(axis=1) < 0.30]
    rppa_abs = [str(a).strip() for a in rppa.index]
    rppa_target_genes = [RPPA_TO_GENE.get(ab, ab.split("_")[0].lstrip("X").upper()) for ab in rppa_abs]
    rppa_sids = set(rppa.columns)

    sid_to_mut = {}
    for mf in sorted(glob.glob(os.path.join(TCGA_MAF_DIR, "TCGA_*_mutect2_maf"))):
        df_m = pd.read_csv(mf, sep="\t", comment="#", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"], low_memory=False)
        df_m["Hugo_Symbol"] = df_m["Hugo_Symbol"].astype(str).str.strip().str.upper()
        df_m = df_m[(df_m["Variant_Classification"] != "Silent") & (df_m["Hugo_Symbol"].isin(driver_set))]
        for sym, bc in zip(df_m["Hugo_Symbol"].values, df_m["Tumor_Sample_Barcode"].astype(str).values):
            s15 = bc[:15]
            if s15 in rppa_sids:
                if s15 not in sid_to_mut:
                    sid_to_mut[s15] = np.zeros(len(driver_genes), dtype=np.float32)
                sid_to_mut[s15][drv_pos[sym]] = 1.0

    tx_l, tm_l, ty_l, tc_l = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(TCGA_RNA_DIR, "*_mrna_tpm.csv"))):
        ctype = os.path.basename(fp).replace("TCGA-", "").replace("_mrna_tpm.csv", "")
        df_r = pd.read_csv(fp, low_memory=False)
        df_r["gene"] = df_r["gene"].astype(str).str.strip().str.upper()
        df_r = df_r[df_r["gene"].isin(top_rna)].drop_duplicates("gene").set_index("gene")
        col_to_s15 = {c: str(c)[:15] for c in df_r.columns if str(c)[:15] in rppa_sids}
        if not col_to_s15:
            continue
        seen = {}
        for oc, s15 in col_to_s15.items():
            if s15 not in seen:
                seen[s15] = oc
        s15s = sorted(seen.keys())
        tx_l.append(row_zscore(np.log2(np.clip(df_r[[seen[s] for s in s15s]].reindex(top_rna).fillna(0.0).values.T, 0.0, None) + 1.0)))
        tm_l.append(np.stack([sid_to_mut.get(s, np.zeros(len(driver_genes), dtype=np.float32)) for s in s15s]))
        ty_l.append(np.nan_to_num(rppa[s15s].values.T.astype(np.float32), nan=0.0))
        tc_l.extend([ctype] * len(s15s))
    X_tcga_r, X_tcga_mut, Y_tcga_rppa, C_tcga = np.vstack(tx_l), np.vstack(tm_l), np.vstack(ty_l), np.array(tc_l)

    # Load CPTAC
    cptac_cancers = sorted(CPTAC_MUT_PREFIX.keys())
    sm, sph, cp_raw = None, None, {}
    for c in cptac_cancers:
        r = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_transcriptomics_bcm_processed_data.csv"), index_col=0)
        p = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_proteomics_bcm_processed_data.csv"), index_col=0)
        ph = pd.read_csv(os.path.join(CPTAC_DIR, f"{c}_phosphoproteomics_bcm_processed_data.csv"), index_col=0)
        r.index = [str(i).strip().upper() for i in r.index]; r = r.groupby(r.index, sort=False).mean()
        p.index = [str(i).strip().upper() for i in p.index]; p = p.groupby(p.index, sort=False).mean()
        ph.index = [str(i).split(":")[0].strip().upper() for i in ph.index]; ph = ph.groupby(ph.index, sort=False).mean()
        mir_p = os.path.join(CPTAC_DIR, f"{c}_miRNA_bcm_processed_data.csv")
        m_df = pd.read_csv(mir_p, index_col=0) if os.path.exists(mir_p) else None
        if m_df is not None:
            m_df.index = [str(i).strip().lower() for i in m_df.index]; m_df = m_df.groupby(m_df.index, sort=False).mean()
            sm = set(m_df.index) if sm is None else (sm & set(m_df.index))
        sph = set(ph.index) if sph is None else (sph & set(ph.index))
        mut_p = os.path.join(CPTAC_MUT_DIR, f"{CPTAC_MUT_PREFIX[c]}_somatic_mutation_gene_level_binary.txt")
        mut_df = pd.read_csv(mut_p, sep="\t", index_col=0)
        mut_df["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in mut_df.index]
        mut_df = mut_df[mut_df["__sym__"].isin(driver_set)].groupby("__sym__", sort=False).max()
        cp_raw[c] = (r, m_df, p, ph, mut_df)

    top_mir = sorted(list(cp_raw["brca"][1].loc[sorted(sm)].var(axis=1).sort_values(ascending=False).index[:350]))
    top_phos = sorted(list(cp_raw["brca"][3].loc[sorted(sph)].var(axis=1).sort_values(ascending=False).index[:800]))

    cx_l, cmut_l, cmir_l, chas_mir_l, cp_l, cph_l, cc_l = [], [], [], [], [], [], []
    for c in cptac_cancers:
        r, m_df, p, ph, mut_df = cp_raw[c]
        sids = sorted(set(r.columns) & set(p.columns) & set(ph.columns))
        cx_l.append(row_zscore(r[sids].reindex(top_rna).fillna(0.0).T.values))
        cp_l.append(row_zscore(p[sids].reindex(top_prot).fillna(0.0).T.values))
        cph_l.append(row_zscore(ph[sids].reindex(top_phos).fillna(0.0).T.values))
        cmut_l.append(mut_df.reindex(index=driver_genes, columns=sids).fillna(0.0).T.values.astype(np.float32))
        if m_df is not None:
            mir_mat = np.zeros((len(sids), len(top_mir)), dtype=np.float32)
            has_m = np.array([s in m_df.columns for s in sids], dtype=bool)
            vp = np.where(has_m)[0]
            if len(vp):
                mir_mat[vp] = row_zscore(m_df[[sids[i] for i in vp]].reindex(top_mir).fillna(0.0).T.values)
            cmir_l.append(mir_mat); chas_mir_l.append(has_m)
        else:
            cmir_l.append(np.zeros((len(sids), len(top_mir)), dtype=np.float32)); chas_mir_l.append(np.zeros(len(sids), dtype=bool))
        cc_l.extend([c.upper()] * len(sids))
    X_cp_r, X_cp_mut, X_cp_mir, has_cp_mir, Y_cp_prot, Y_cp_phos, C_cp = (
        np.vstack(cx_l), np.vstack(cmut_l), np.vstack(cmir_l), np.concatenate(chas_mir_l), np.vstack(cp_l), np.vstack(cph_l), np.array(cc_l)
    )

    # Graphs & Masks
    A_tf, A_gp, M_rppa, M_prot, M_phos = build_graphs_and_mut_masks(top_rna, driver_genes, rppa_target_genes, top_prot, top_phos)
    tf_list = [tf for tf in CORE_TFS if tf in set(top_rna)]
    rppa_cis = [rna_pos.get(g, -1) for g in rppa_target_genes]
    prot_cis = [rna_pos.get(g, -1) for g in top_prot]
    phos_cis = [rna_pos.get(g, -1) for g in top_phos]

    tr_t = ~np.isin(C_tcga, list(UNSEEN_CANCERS))
    tr_c = ~np.isin(C_cp, list(UNSEEN_CANCERS))

    net = SparseReg5OmicsPathwayGNN(
        len(top_rna), len(driver_genes), len(top_mir),
        rppa_cis, prot_cis, phos_cis, A_tf, A_gp, M_rppa, M_prot, M_phos
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1.2e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()

    l1 = DataLoader(TensorDataset(torch.tensor(X_tcga_r[tr_t]), torch.tensor(X_tcga_mut[tr_t]), torch.tensor(Y_tcga_rppa[tr_t])), batch_size=256, shuffle=True)
    l2 = DataLoader(TensorDataset(torch.tensor(X_cp_r[tr_c]), torch.tensor(X_cp_mut[tr_c]), torch.tensor(X_cp_mir[tr_c]), torch.tensor(has_cp_mir[tr_c]), torch.tensor(Y_cp_prot[tr_c]), torch.tensor(Y_cp_phos[tr_c])), batch_size=128, shuffle=True)

    net.train()
    for ep in range(40):
        for br, bm, by in l1:
            br, bm, by = br.to(DEVICE), bm.to(DEVICE), by.to(DEVICE)
            bm_in = None if np.random.rand() < 0.2 else bm
            opt.zero_grad()
            a_r, a_t, _ = net.get_gates()
            l1_pen = 0.015 * (a_r.mean() + a_t.mean())
            loss = mse_fn(net.forward_rppa(br, bm_in, None), by) + l1_pen + 5e-4 * (net.W_mut_rppa ** 2).mean()
            loss.backward(); opt.step()
        for br, bm, bmi, bhasm, bp, bph in l2:
            br, bm, bmi, bp, bph = br.to(DEVICE), bm.to(DEVICE), bmi.to(DEVICE), bp.to(DEVICE), bph.to(DEVICE)
            bm_in = None if np.random.rand() < 0.2 else bm
            bmi_in = None if (not bhasm.any() or np.random.rand() < 0.25) else (bmi * bhasm.to(DEVICE).float().unsqueeze(-1))
            opt.zero_grad()
            _, _, a_m = net.get_gates()
            loss = mse_fn(net.forward_prot(br, bm_in, bmi_in), bp) + mse_fn(net.forward_phos(br, bm_in, bmi_in), bph) + 0.02 * a_m.mean()
            loss.backward(); opt.step()

    net.eval()
    results = []
    with torch.no_grad():
        for c in sorted(UNSEEN_CANCERS):
            mt = (C_tcga == c)
            xt_r = torch.tensor(X_tcga_r[mt], device=DEVICE)
            xt_m = torch.tensor(X_tcga_mut[mt], device=DEVICE)
            yp = net.forward_rppa(xt_r, xt_m, None).cpu().numpy()
            rs = [pearsonr(yp[:, j], Y_tcga_rppa[mt, j])[0] for j in range(len(rppa_abs)) if np.std(Y_tcga_rppa[mt, j]) > 1e-6]
            prs = [pearsonr(yp[:, j], Y_tcga_rppa[mt, j])[0] for j, ab in enumerate(rppa_abs) if ("_p" in ab and np.std(Y_tcga_rppa[mt, j]) > 1e-6)]
            results.append({"cancer": c, "platform": "TCGA_RPPA_All", "median_r": float(np.median(rs))})
            results.append({"cancer": c, "platform": "TCGA_RPPA_Phospho", "median_r": float(np.median(prs))})
            print(f"  [BLIND TCGA-{c:4s}] SparseReg-GNN RPPA All Med r={np.median(rs):.4f} | Phospho Med r={np.median(prs):.4f}")

            mc = (C_cp == c)
            xc_r = torch.tensor(X_cp_r[mc], device=DEVICE)
            xc_m = torch.tensor(X_cp_mut[mc], device=DEVICE)
            xc_mi = torch.tensor(X_cp_mir[mc], device=DEVICE) if has_cp_mir[mc].any() else None
            yp_p = net.forward_prot(xc_r, xc_m, xc_mi).cpu().numpy()
            yp_ph = net.forward_phos(xc_r, xc_m, xc_mi).cpu().numpy()
            rp = float(np.median([pearsonr(yp_p[:, j], Y_cp_prot[mc, j])[0] for j in range(len(top_prot)) if np.std(Y_cp_prot[mc, j]) > 1e-6]))
            rph = float(np.median([pearsonr(yp_ph[:, j], Y_cp_phos[mc, j])[0] for j in range(len(top_phos)) if np.std(Y_cp_phos[mc, j]) > 1e-6]))
            results.append({"cancer": c, "platform": "CPTAC_MS_Proteome", "median_r": rp})
            results.append({"cancer": c, "platform": "CPTAC_MS_Phospho", "median_r": rph})
            print(f"  [BLIND CPTAC-{c:4s}] SparseReg-GNN MS Prot Med r={rp:.4f} | MS Phospho Med r={rph:.4f}")

        # Extract learned importance gates & top mutation->target regulatory weights
        a_r, a_t, a_m = [t.cpu().numpy() for t in net.get_gates()]
        w_mut_r = (net.W_mut_rppa * net.M_rppa).cpu().numpy()

    imp_rows = []
    for g, w in zip(top_rna, a_r):
        imp_rows.append({"regulator_type": "RNA_Gene", "name": g, "learned_weight": float(w)})
    for tf, w in zip(tf_list, a_t):
        imp_rows.append({"regulator_type": "TF_Hub", "name": tf, "learned_weight": float(w)})
    for mi, w in zip(top_mir, a_m):
        imp_rows.append({"regulator_type": "miRNA", "name": mi, "learned_weight": float(w)})
    for i, d in enumerate(driver_genes):
        for j, ab in enumerate(rppa_abs):
            if abs(w_mut_r[i, j]) > 0.05:
                imp_rows.append({"regulator_type": "Mut->RPPA_Edge", "name": f"{d} -> {ab}", "learned_weight": float(w_mut_r[i, j])})

    imp_df = pd.DataFrame(imp_rows).sort_values("learned_weight", ascending=False)
    imp_df.to_csv(OUT_IMP_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Top 10 Learned Master TFs ===")
    print(imp_df[imp_df["regulator_type"] == "TF_Hub"].head(10).to_string(index=False))
    print("\n=== Top 10 Learned Key miRNAs ===")
    print(imp_df[imp_df["regulator_type"] == "miRNA"].head(10).to_string(index=False))
    print("\n=== Top 10 Learned Mut -> RPPA Protein/Phospho Regulatory Edges ===")
    ed_df = imp_df[imp_df["regulator_type"] == "Mut->RPPA_Edge"].copy()
    ed_df["abs_w"] = ed_df["learned_weight"].abs()
    print(ed_df.sort_values("abs_w", ascending=False).head(10)[["name", "learned_weight"]].to_string(index=False))

    # Plot 3-panel learned regulatory importance
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=180)
    for ax, rtype, title, col in [
        (axes[0], "TF_Hub", "Top Learned Master TFs\n(BioGRID Regulon Gate)", "#2980b9"),
        (axes[1], "miRNA", "Top Learned Key miRNAs\n(L1-Sparse Importance Gate)", "#27ae60"),
    ]:
        sub = imp_df[imp_df["regulator_type"] == rtype].head(12)
        yp = np.arange(len(sub))
        ax.barh(yp, sub["learned_weight"], color=col, edgecolor="black", linewidth=0.5)
        ax.set_yticks(yp); ax.set_yticklabels(sub["name"], fontsize=8.5); ax.invert_yaxis()
        ax.set_xlabel("Learned Importance Gate (0-1)")
        ax.set_title(title, fontsize=10, fontweight="bold")

    top_ed = ed_df.sort_values("abs_w", ascending=False).head(12)
    yp = np.arange(len(top_ed))
    cols_e = ["#e74c3c" if v > 0 else "#3498db" for v in top_ed["learned_weight"]]
    axes[2].barh(yp, top_ed["learned_weight"], color=cols_e, edgecolor="black", linewidth=0.5)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_yticks(yp); axes[2].set_yticklabels(top_ed["name"], fontsize=8); axes[2].invert_yaxis()
    axes[2].set_xlabel("Learned Regulatory Weight (Red=Up, Blue=Down)")
    axes[2].set_title("Top Driver Mut -> Protein/Phospho\nLearned Causal Edges", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp28 results to {OUT_CSV}, {OUT_IMP_CSV}, & {OUT_PNG}")


if __name__ == "__main__":
    main()
