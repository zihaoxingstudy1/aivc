import os
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPTAC_DIR = r"D:\data\CPTAC\cptac_processed_data"
BIOGRID_PATH = r"D:\data\regulation\BIOGRID-ALL-4.4.211.tab3.txt"

CANCERS_9 = ["brca", "ccrcc", "coad", "gbm", "hnscc", "lscc", "luad", "pdac", "ucec"]
TRAIN_CANCERS = ["brca", "ccrcc", "coad", "gbm", "hnscc", "lscc", "ucec"]
TEST_CANCERS = ["luad", "pdac"]

OUT_CSV = r"D:\data\AIVC\test_results\exp25_tf_mirna_reggnn_results.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp25_tf_mirna_reggnn_plot.png"

CORE_TFS = [
    "TP53", "MYC", "E2F1", "E2F3", "HIF1A", "STAT3", "STAT1", "ESR1", "FOXA1", "FOXM1",
    "JUN", "FOS", "RELA", "NFKB1", "SP1", "ETS1", "CTNNB1", "SMAD3", "SMAD4", "AR",
    "SOX2", "KLF4", "BRCA1", "YAP1", "TEAD1", "CEBPB", "ATF4", "NRF2", "NFE2L2", "CREB1",
    "TWIST1", "SNAI1", "ZEB1", "REST", "YY1", "MAX", "EGR1", "IRF1", "GATA3", "RUNX1",
    "PAX8", "NKX2-1", "HNF4A", "PPARG", "RXRA", "SRF", "ELK1", "MITF", "FOXO1", "FOXO3",
]


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_4omics_cancer(cancer: str):
    r = pd.read_csv(os.path.join(CPTAC_DIR, f"{cancer}_transcriptomics_bcm_processed_data.csv"), index_col=0)
    m = pd.read_csv(os.path.join(CPTAC_DIR, f"{cancer}_miRNA_bcm_processed_data.csv"), index_col=0)
    p = pd.read_csv(os.path.join(CPTAC_DIR, f"{cancer}_proteomics_bcm_processed_data.csv"), index_col=0)
    ph = pd.read_csv(os.path.join(CPTAC_DIR, f"{cancer}_phosphoproteomics_bcm_processed_data.csv"), index_col=0)

    for df in (r, p):
        df.index = [str(i).strip().upper() for i in df.index]
    r = r.groupby(r.index, sort=False).mean()
    p = p.groupby(p.index, sort=False).mean()
    m.index = [str(i).strip().lower() for i in m.index]
    m = m.groupby(m.index, sort=False).mean()
    ph.index = [str(i).split(":")[0].strip().upper() for i in ph.index]
    ph = ph.groupby(ph.index, sort=False).mean()

    common_sids = sorted(set(r.columns) & set(m.columns) & set(p.columns) & set(ph.columns))
    return r[common_sids], m[common_sids], p[common_sids], ph[common_sids], common_sids


def build_biogrid_tf_and_kegg_graphs(rna_genes: list):
    """Build (1) TF-to-Gene BioGRID Regulon adjacency and (2) KEGG Gene-to-Pathway adjacency."""
    gene_set = set(rna_genes)
    gene_to_idx = {g: i for i, g in enumerate(rna_genes)}

    # 1. Extract Human BioGRID edges involving CORE_TFS
    tf_list = [tf for tf in CORE_TFS if tf in gene_set]
    tf_set = set(tf_list)
    tf_to_idx = {tf: j for j, tf in enumerate(tf_list)}
    A_tf = np.zeros((len(rna_genes), len(tf_list)), dtype=np.float32)

    print("Extracting Human BioGRID TF-Gene regulatory edges...")
    chunk_iter = pd.read_csv(
        BIOGRID_PATH,
        sep="\t",
        usecols=["Official Symbol Interactor A", "Official Symbol Interactor B", "Organism ID Interactor A", "Organism ID Interactor B"],
        chunksize=250000,
        low_memory=False,
    )
    edge_count = 0
    for chunk in chunk_iter:
        hu = chunk[(chunk["Organism ID Interactor A"] == 9606) & (chunk["Organism ID Interactor B"] == 9606)]
        a_sym = hu["Official Symbol Interactor A"].astype(str).str.upper().values
        b_sym = hu["Official Symbol Interactor B"].astype(str).str.upper().values
        for ga, gb in zip(a_sym, b_sym):
            if ga in tf_set and gb in gene_set:
                A_tf[gene_to_idx[gb], tf_to_idx[ga]] = 1.0
                edge_count += 1
            if gb in tf_set and ga in gene_set:
                A_tf[gene_to_idx[ga], tf_to_idx[gb]] = 1.0
                edge_count += 1

    # Add self-loop for each TF
    for tf in tf_list:
        A_tf[gene_to_idx[tf], tf_to_idx[tf]] = 1.0
    col_deg_tf = np.sqrt(np.maximum(A_tf.sum(axis=0, keepdims=True), 1.0))
    A_tf_norm = (A_tf / col_deg_tf).astype(np.float32)

    # 2. KEGG Signaling Pathways
    kegg = gp.get_library("KEGG_2021_Human")
    priority_kw = ("signaling", "cancer", "cell cycle", "apoptosis", "p53", "dna replication", "proteasome", "glycolysis")
    cand = []
    for pw_name, pw_genes in kegg.items():
        ov = [g.upper() for g in pw_genes if g.upper() in gene_set]
        if len(ov) >= 12:
            sc = (1000 if any(k in pw_name.lower() for k in priority_kw) else 0) + len(ov)
            cand.append((sc, pw_name, ov))
    cand.sort(reverse=True)
    sel = cand[:45]
    A_gp = np.zeros((len(rna_genes), len(sel)), dtype=np.float32)
    for j, (_, _, ov) in enumerate(sel):
        for g in ov:
            A_gp[gene_to_idx[g], j] = 1.0
    A_gp_norm = (A_gp / np.sqrt(np.maximum(A_gp.sum(axis=0, keepdims=True), 1.0))).astype(np.float32)

    print(f"Built BioGRID TF Graph: {len(tf_list)} TF Hubs ({int(A_tf.sum())} unique TF-Gene edges) | KEGG Graph: {len(sel)} Pathway Hubs ({int(A_gp.sum())} edges)")
    return A_tf_norm, A_gp_norm, tf_list


class HeteroRegGNN(nn.Module):
    """
    4-Layer Heterogeneous Regulatory GNN:
      - Layer 1: BioGRID TF-Regulon Activity Graph (RNA -> TF Activity)
      - Layer 2: miRNA Post-Transcriptional Silencing Encoder (with Modality Dropout)
      - Layer 3: KEGG Pathway Hub Graph
      - Layer 4: Cis-Residual + Multi-Omics Trans Decoders (Proteome + Phosphoproteome)
    """
    def __init__(
        self,
        n_rna: int,
        n_mir: int,
        prot_cis_idx: list,
        phos_cis_idx: list,
        A_tf: np.ndarray,
        A_gp: np.ndarray,
        use_tf: bool = True,
        use_mirna: bool = True,
        hid: int = 160,
    ):
        super().__init__()
        self.use_tf = use_tf
        self.use_mirna = use_mirna
        self.register_buffer("A_tf", torch.tensor(A_tf, dtype=torch.float32))
        self.register_buffer("A_gp", torch.tensor(A_gp, dtype=torch.float32))
        self.register_buffer("prot_cis_idx", torch.tensor(prot_cis_idx, dtype=torch.long))
        self.register_buffer("phos_cis_idx", torch.tensor(phos_cis_idx, dtype=torch.long))

        n_prot = len(prot_cis_idx)
        n_phos = len(phos_cis_idx)
        self.prot_cis_w = nn.Parameter(torch.ones(n_prot) * 0.5)
        self.phos_cis_w = nn.Parameter(torch.ones(n_phos) * 0.25)

        self.pw_enc = nn.Sequential(nn.Linear(A_gp.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.tf_enc = nn.Sequential(nn.Linear(A_tf.shape[1], hid), nn.LayerNorm(hid), nn.GELU())
        self.mir_enc = nn.Sequential(nn.Linear(n_mir, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.rna_glob = nn.Sequential(nn.Linear(n_rna, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.3))

        in_dim = hid * 2 + (hid if use_tf else 0) + (hid if use_mirna else 0)
        self.fuse = nn.Sequential(nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.prot_dec = nn.Linear(hid, n_prot)
        self.phos_dec = nn.Linear(hid, n_phos)
        nn.init.zeros_(self.prot_dec.weight)
        nn.init.zeros_(self.phos_dec.weight)

    def forward(self, x_rna: torch.Tensor, x_mir: torch.Tensor = None):
        feats = [self.pw_enc(x_rna @ self.A_gp), self.rna_glob(x_rna)]
        if self.use_tf:
            feats.append(self.tf_enc(x_rna @ self.A_tf))
        if self.use_mirna:
            if x_mir is None:
                feats.append(torch.zeros_like(feats[0]))
            else:
                feats.append(self.mir_enc(x_mir))
        h = self.fuse(torch.cat(feats, dim=-1))

        vp = (self.prot_cis_idx >= 0).float()
        sp = torch.clamp(self.prot_cis_idx, min=0)
        vph = (self.phos_cis_idx >= 0).float()
        sph = torch.clamp(self.phos_cis_idx, min=0)

        p_out = x_rna[:, sp] * vp * self.prot_cis_w + 0.35 * self.prot_dec(h)
        ph_out = x_rna[:, sph] * vph * self.phos_cis_w + 0.35 * self.phos_dec(h)
        return p_out, ph_out


def main():
    print(f"=== Exp25: 4-Layer Regulatory GNN (TF-BioGRID + miRNA + KEGG + Proteome/Phospho) ({DEVICE}) ===")
    loaded = {}
    sr, sm, sp, sph = None, None, None, None
    for c in CANCERS_9:
        r, m, p, ph, sids = load_4omics_cancer(c)
        loaded[c] = (r, m, p, ph, sids)
        sr = set(r.index) if sr is None else (sr & set(r.index))
        sm = set(m.index) if sm is None else (sm & set(m.index))
        sp = set(p.index) if sp is None else (sp & set(p.index))
        sph = set(ph.index) if sph is None else (sph & set(ph.index))
        print(f"  {c:6s}: 4-Omics Paired Samples={len(sids):3d} | miRNAs={len(m)} | Prot={len(p)} | Phos={len(ph)}")

    brca_r, brca_m, brca_p, brca_ph, _ = loaded["brca"]
    top_rna = list(brca_r.loc[sorted(sr)].var(axis=1).sort_values(ascending=False).index[:2000])
    top_mir = sorted(list(brca_m.loc[sorted(sm)].var(axis=1).sort_values(ascending=False).index[:350]))
    top_prot = sorted(list(brca_p.loc[sorted(sp)].var(axis=1).sort_values(ascending=False).index[:1200]))
    top_phos = sorted(list(brca_ph.loc[sorted(sph)].var(axis=1).sort_values(ascending=False).index[:800]))

    for g in CORE_TFS + top_prot + top_phos:
        if g in sr and g not in top_rna:
            top_rna.append(g)
    top_rna = sorted(top_rna)
    rna_pos = {g: i for i, g in enumerate(top_rna)}
    prot_cis_idx = [rna_pos.get(g, -1) for g in top_prot]
    phos_cis_idx = [rna_pos.get(g, -1) for g in top_phos]

    A_tf, A_gp, tf_list = build_biogrid_tf_and_kegg_graphs(top_rna)

    def build_split(c_list):
        xr, xm, xp, xph, cn = [], [], [], [], []
        for c in c_list:
            r, m, p, ph, sids = loaded[c]
            xr.append(row_zscore(r.loc[top_rna].T.fillna(0.0).values))
            xm.append(row_zscore(m.loc[top_mir].T.fillna(0.0).values))
            xp.append(row_zscore(p.loc[top_prot].T.fillna(0.0).values))
            xph.append(row_zscore(ph.loc[top_phos].T.fillna(0.0).values))
            cn.extend([c.upper()] * len(sids))
        return np.vstack(xr), np.vstack(xm), np.vstack(xp), np.vstack(xph), np.array(cn)

    X_tr_r, X_tr_m, Y_tr_p, Y_tr_ph, _ = build_split(TRAIN_CANCERS)
    X_te_r, X_te_m, Y_te_p, Y_te_ph, te_c = build_split(TEST_CANCERS)
    print(f"Train (7 cancers): {len(X_tr_r)} | Blind Test (LUAD + PDAC): {len(X_te_r)}")

    configs = [
        ("1_KEGG_PathwayGNN_RNAonly", False, False),
        ("2_TF_BioGRID+KEGG_RNAonly", True, False),
        ("3_Full_4Layer_TF+miRNA+KEGG", True, True),
    ]
    results = []
    for cfg_name, use_tf, use_mir in configs:
        torch.manual_seed(SEED)
        net = HeteroRegGNN(len(top_rna), len(top_mir), prot_cis_idx, phos_cis_idx, A_tf, A_gp, use_tf=use_tf, use_mirna=use_mir).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.tensor(X_tr_r), torch.tensor(X_tr_m), torch.tensor(Y_tr_p), torch.tensor(Y_tr_ph)), batch_size=64, shuffle=True)
        net.train()
        for ep in range(40):
            for br, bm, bp, bph in loader:
                br, bm, bp, bph = br.to(DEVICE), bm.to(DEVICE), bp.to(DEVICE), bph.to(DEVICE)
                # 20% Modality Dropout on miRNA during training so RNA-only fallback stays intact
                bm_in = None if (use_mir and np.random.rand() < 0.2) else bm
                opt.zero_grad()
                pp, pph = net(br, bm_in)
                loss = loss_fn(pp, bp) + loss_fn(pph, bph)
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            for c_t in [c.upper() for c in TEST_CANCERS]:
                mask = (te_c == c_t)
                xr_t = torch.tensor(X_te_r[mask], device=DEVICE)
                xm_t = torch.tensor(X_te_m[mask], device=DEVICE) if use_mir else None
                pp, pph = net(xr_t, xm_t)
                pp, pph = pp.cpu().numpy(), pph.cpu().numpy()

                r_prot = [pearsonr(pp[:, j], Y_te_p[mask, j])[0] for j in range(len(top_prot)) if np.std(Y_te_p[mask, j]) > 1e-6]
                r_phos = [pearsonr(pph[:, j], Y_te_ph[mask, j])[0] for j in range(len(top_phos)) if np.std(Y_te_ph[mask, j]) > 1e-6]
                med_p, med_ph = float(np.median(r_prot)), float(np.median(r_phos))
                print(f"  [{cfg_name:28s}] Blind {c_t:4s} | Proteome Med r={med_p:.4f} | Phospho Med r={med_ph:.4f}")
                results.append({"model": cfg_name, "unseen_cancer": c_t, "modality": "Proteome", "median_r": med_p})
                results.append({"model": cfg_name, "unseen_cancer": c_t, "modality": "Phosphoproteome", "median_r": med_ph})

                if use_mir:
                    # Also verify Modality Dropout fallback when miRNA is NOT provided at test time!
                    pp_fb, pph_fb = net(xr_t, None)
                    pp_fb, pph_fb = pp_fb.cpu().numpy(), pph_fb.cpu().numpy()
                    rp_fb = float(np.median([pearsonr(pp_fb[:, j], Y_te_p[mask, j])[0] for j in range(len(top_prot)) if np.std(Y_te_p[mask, j]) > 1e-6]))
                    rph_fb = float(np.median([pearsonr(pph_fb[:, j], Y_te_ph[mask, j])[0] for j in range(len(top_phos)) if np.std(Y_te_ph[mask, j]) > 1e-6]))
                    print(f"    -> Fallback (miRNA omitted)    Blind {c_t:4s} | Proteome Med r={rp_fb:.4f} | Phospho Med r={rph_fb:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    for idx_m, mod in enumerate(["Proteome", "Phosphoproteome"]):
        sub = res_df[res_df["modality"] == mod]
        piv = sub.pivot(index="unseen_cancer", columns="model", values="median_r")
        piv.plot(kind="bar", ax=axes[idx_m], width=0.72, edgecolor="black", linewidth=0.5)
        axes[idx_m].set_title(f"4-Layer Regulatory GNN Ablation ({mod})\n(Blind Unseen Cancers: LUAD & PDAC)", fontsize=10.5, fontweight="bold")
        axes[idx_m].set_ylabel("Median Cross-Patient Pearson (r)")
        axes[idx_m].tick_params(axis="x", rotation=0)
        axes[idx_m].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved Exp25 results to {OUT_CSV} & {OUT_PNG}")


if __name__ == "__main__":
    main()
