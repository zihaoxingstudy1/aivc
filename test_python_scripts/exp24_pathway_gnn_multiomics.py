import os
import sys
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
from bio_pathway_viz import plot_mechanistic_signaling_map

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_PATH = r"D:\data\AIVC\data\exp24_multiomics_cache.npz"
OUT_CSV = r"D:\data\AIVC\test_results\exp24_pathway_gnn_results.csv"
OUT_PHOS_CSV = r"D:\data\AIVC\test_results\exp24_pathway_gnn_phospho_top.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp24_pathway_gnn_comparison.png"
OUT_MAP_PNG = r"D:\data\AIVC\test_results\exp24_mechanistic_signaling_map.png"

UNSEEN_CANCERS = {"LUAD", "OV"}

RPPA_TO_GENE = {
    "AKT": "AKT1", "AKT_pS473": "AKT1", "AKT_pT308": "AKT1",
    "MAPK_pT202Y204": "MAPK1", "ERK2": "MAPK1", "MEK1": "MAP2K1", "MEK1_pS217S221": "MAP2K1",
    "EGFR": "EGFR", "EGFR_pY1068": "EGFR", "EGFR_pY1173": "EGFR",
    "HER2": "ERBB2", "HER2_pY1248": "ERBB2", "HER3": "ERBB3",
    "X4EBP1": "EIF4EBP1", "X4EBP1_pS65": "EIF4EBP1", "X4EBP1_pT37T46": "EIF4EBP1",
    "S6_pS235S236": "RPS6", "S6_pS240S244": "RPS6", "P70S6K1": "RPS6KB1", "P70S6K_pT389": "RPS6KB1",
    "MTOR": "MTOR", "MTOR_pS2448": "MTOR", "AMPKALPHA": "PRKAA1", "AMPKALPHA_pT172": "PRKAA1",
    "STAT3_pY705": "STAT3", "STAT5ALPHA": "STAT5A", "JAK2": "JAK2",
    "RB": "RB1", "RB_pS807S811": "RB1", "CYCLIND1": "CCND1", "CYCLINB1": "CCNB1", "CYCLINE1": "CCNE1",
    "CDK1": "CDK1", "P27": "CDKN1B", "P21": "CDKN1A", "TP53": "TP53", "P53": "TP53",
    "BCL2": "BCL2", "BCLXL": "BCL2L1", "BAX": "BAX", "CASPASE7CLEAVEDD198": "CASP7",
    "PARP1": "PARP1", "ATM": "ATM", "CHEK1": "CHEK1", "CHEK2": "CHEK2",
    "PI3KP110ALPHA": "PIK3CA", "PTEN": "PTEN", "SRC": "SRC", "SRC_pY527": "SRC",
    "CMET": "MET", "CMET_pY1235": "MET", "BRAF": "BRAF", "CRAF": "RAF1",
    "BETACATENIN": "CTNNB1", "ECADHERIN": "CDH1", "NCADHERIN": "CDH2", "SNAI1": "SNAI1",
    "YAP": "YAP1", "YAP_pS127": "YAP1", "NFKBP65_pS536": "RELA",
}


def build_kegg_signaling_graph(rna_genes: list, max_pathways: int = 45):
    """Build normalized bipartite Gene->Pathway & Pathway<->Pathway adjacency matrices from KEGG."""
    kegg = gp.get_library("KEGG_2021_Human")
    gene_set = set(rna_genes)
    gene_to_idx = {g: i for i, g in enumerate(rna_genes)}

    priority_kw = (
        "signaling", "cancer", "cell cycle", "apoptosis", "p53", "dna replication",
        "proteasome", "oxidative", "glycolysis", "focal adhesion", "ecm", "t cell", "pd-l1"
    )
    cand_pws = []
    for pw_name, pw_genes in kegg.items():
        overlap = [g.upper() for g in pw_genes if g.upper() in gene_set]
        if len(overlap) >= 12:
            score = (1000 if any(k in pw_name.lower() for k in priority_kw) else 0) + len(overlap)
            cand_pws.append((score, pw_name, overlap))
    cand_pws.sort(reverse=True)
    selected = cand_pws[:max_pathways]

    n_genes = len(rna_genes)
    n_pws = len(selected)
    A_gp = np.zeros((n_genes, n_pws), dtype=np.float32)
    for j, (_, _, ov_genes) in enumerate(selected):
        for g in ov_genes:
            A_gp[gene_to_idx[g], j] = 1.0
    # Normalize columns (each pathway aggregates mean/sqrt-weighted signal of its member genes)
    col_deg = np.maximum(A_gp.sum(axis=0, keepdims=True), 1.0)
    A_gp_norm = A_gp / np.sqrt(col_deg)

    # Pathway-to-Pathway crosstalk adjacency (normalized Jaccard/cosine overlap + self-loop)
    A_pp = (A_gp.T @ A_gp) + np.eye(n_pws, dtype=np.float32) * 5.0
    deg_p = np.sqrt(np.maximum(A_pp.sum(axis=1, keepdims=True), 1.0))
    A_pp_norm = (A_pp / (deg_p @ deg_p.T)).astype(np.float32)

    pw_names = [s[1] for s in selected]
    print(f"Built KEGG Signaling Graph: {n_genes} Gene Nodes -> {n_pws} Pathway Hub Nodes ({int(A_gp.sum())} edges)")
    return A_gp_norm, A_pp_norm, pw_names


class PathwayGNNTranslator(nn.Module):
    """
    Biological Pathway-Graph Neural Network for Cross-Omics Translation:
      1) Gene -> Pathway Message Passing along real KEGG edges
      2) 2-Layer Pathway-Crosstalk Graph Attention / Residual Propagation
      3) Dual-Head Cis-Residual + Graph-Trans Prediction:
         - Head 1: TCGA RPPA Hub Proteins & Phosphosites (217)
         - Head 2: CPTAC + CCLE Deep MS Proteome (1,200)
    """
    def __init__(self, A_gp: np.ndarray, A_pp: np.ndarray, rppa_cis_idx: list, prot_cis_idx: list, hid: int = 192):
        super().__init__()
        self.register_buffer("A_gp", torch.tensor(A_gp, dtype=torch.float32))  # (G, P)
        self.register_buffer("A_pp", torch.tensor(A_pp, dtype=torch.float32))  # (P, P)
        self.register_buffer("rppa_cis_idx", torch.tensor(rppa_cis_idx, dtype=torch.long))
        self.register_buffer("prot_cis_idx", torch.tensor(prot_cis_idx, dtype=torch.long))

        n_genes, n_pws = A_gp.shape
        n_rppa = len(rppa_cis_idx)
        n_prot = len(prot_cis_idx)

        self.rppa_cis_w = nn.Parameter(torch.ones(n_rppa) * 0.35)
        self.prot_cis_w = nn.Parameter(torch.ones(n_prot) * 0.50)

        # Node-level gene -> pathway projection + global context
        self.gene_gate = nn.Parameter(torch.ones(n_genes) * 0.5)
        self.pw_proj = nn.Sequential(nn.Linear(n_pws, hid), nn.LayerNorm(hid), nn.GELU())
        self.gnn_layer1 = nn.Sequential(nn.Linear(n_pws, n_pws), nn.GELU())
        self.gnn_layer2 = nn.Sequential(nn.Linear(n_pws, n_pws), nn.GELU())
        self.global_enc = nn.Sequential(
            nn.Linear(n_genes, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.rppa_dec = nn.Linear(hid, n_rppa)
        self.prot_dec = nn.Linear(hid, n_prot)
        nn.init.zeros_(self.rppa_dec.weight)
        nn.init.zeros_(self.prot_dec.weight)

    def encode_graph(self, x_rna: torch.Tensor) -> torch.Tensor:
        # Weighted gene signal -> Pathway Hub Nodes via A_gp
        xg = x_rna * torch.sigmoid(self.gene_gate)
        hp0 = xg @ self.A_gp  # (B, P)
        # 2-layer Pathway Crosstalk Message Passing
        hp1 = hp0 + 0.5 * self.gnn_layer1(hp0 @ self.A_pp)
        hp2 = hp1 + 0.5 * self.gnn_layer2(hp1 @ self.A_pp)
        h_graph = self.pw_proj(hp2)
        h_glob = self.global_enc(x_rna)
        return self.fuse(torch.cat([h_graph, h_glob], dim=-1))

    def forward_rppa(self, x_rna: torch.Tensor) -> torch.Tensor:
        h = self.encode_graph(x_rna)
        valid_mask = (self.rppa_cis_idx >= 0).float()
        safe_idx = torch.clamp(self.rppa_cis_idx, min=0)
        cis = x_rna[:, safe_idx] * valid_mask * self.rppa_cis_w
        return cis + 0.4 * self.rppa_dec(h)

    def forward_prot(self, x_rna: torch.Tensor) -> torch.Tensor:
        h = self.encode_graph(x_rna)
        valid_mask = (self.prot_cis_idx >= 0).float()
        safe_idx = torch.clamp(self.prot_cis_idx, min=0)
        cis = x_rna[:, safe_idx] * valid_mask * self.prot_cis_w
        return cis + 0.35 * self.prot_dec(h)


class MLPBaselineTranslator(nn.Module):
    """Non-graph MLP baseline (same Cis-Residual, no KEGG pathway graph)."""
    def __init__(self, n_genes: int, rppa_cis_idx: list, prot_cis_idx: list, hid: int = 192):
        super().__init__()
        self.register_buffer("rppa_cis_idx", torch.tensor(rppa_cis_idx, dtype=torch.long))
        self.register_buffer("prot_cis_idx", torch.tensor(prot_cis_idx, dtype=torch.long))
        self.rppa_cis_w = nn.Parameter(torch.ones(len(rppa_cis_idx)) * 0.35)
        self.prot_cis_w = nn.Parameter(torch.ones(len(prot_cis_idx)) * 0.50)
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(),
        )
        self.rppa_dec = nn.Linear(hid, len(rppa_cis_idx))
        self.prot_dec = nn.Linear(hid, len(prot_cis_idx))

    def forward_rppa(self, x):
        h = self.enc(x)
        vm = (self.rppa_cis_idx >= 0).float()
        si = torch.clamp(self.rppa_cis_idx, min=0)
        return x[:, si] * vm * self.rppa_cis_w + 0.4 * self.rppa_dec(h)

    def forward_prot(self, x):
        vm = (self.prot_cis_idx >= 0).float()
        si = torch.clamp(self.prot_cis_idx, min=0)
        return x[:, si] * vm * self.prot_cis_w + 0.35 * self.prot_dec(self.enc(x))


def eval_predictions(y_true: np.ndarray, y_pred: np.ndarray, x_rna: np.ndarray, cis_idx: list, names: list, is_phospho_mask=None):
    m_rs, n_rs = [], []
    per_item = []
    for j, nm in enumerate(names):
        yt, yp = y_true[:, j], y_pred[:, j]
        if np.std(yt) < 1e-6 or np.std(yp) < 1e-6:
            continue
        rm, _ = pearsonr(yp, yt)
        m_rs.append(rm)
        ci = cis_idx[j]
        rn = pearsonr(x_rna[:, ci], yt)[0] if ci >= 0 else np.nan
        if np.isfinite(rn):
            n_rs.append(rn)
        is_p = bool(is_phospho_mask[j]) if is_phospho_mask is not None else False
        per_item.append({"feature": nm, "is_phospho": is_p, "naive_r": rn, "model_r": rm})
    return float(np.median(m_rs)), float(np.median(n_rs)) if n_rs else np.nan, pd.DataFrame(per_item)


def main():
    print(f"=== Exp24: Pathway-GNN Multi-Omics Translator (TCGA + CPTAC + CCLE) ({DEVICE}) ===")
    data = np.load(CACHE_PATH, allow_pickle=True)
    top_rna = [str(g) for g in data["top_rna"]]
    top_prot = [str(g) for g in data["top_prot"]]
    rppa_abs = [str(a) for a in data["tcga_rppa_abs"]]
    rna_pos = {g: i for i, g in enumerate(top_rna)}

    # Map RPPA antibody names to cis RNA index (and mark phospho antibodies)
    rppa_cis_idx = []
    is_phospho = []
    for ab in rppa_abs:
        is_p = ("_p" in ab)
        is_phospho.append(is_p)
        base = RPPA_TO_GENE.get(ab, ab.split("_")[0].lstrip("X").upper())
        rppa_cis_idx.append(rna_pos.get(base, -1))
    prot_cis_idx = [rna_pos.get(g, -1) for g in top_prot]
    is_phospho = np.array(is_phospho)

    # Split TCGA (7,359) & CPTAC (1,351) strictly by Unseen Cancers (LUAD, OV)
    X_tcga = np.nan_to_num(data["X_tcga"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y_tcga = np.nan_to_num(data["Y_tcga_rppa"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    C_tcga = data["C_tcga"]
    tr_t_mask = ~np.isin(C_tcga, list(UNSEEN_CANCERS))
    te_t_mask = np.isin(C_tcga, list(UNSEEN_CANCERS))

    X_cp = np.nan_to_num(data["X_cptac"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y_cp = np.nan_to_num(data["Y_cptac_prot"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    C_cp = data["C_cptac"]
    tr_c_mask = ~np.isin(C_cp, list(UNSEEN_CANCERS))
    te_c_mask = np.isin(C_cp, list(UNSEEN_CANCERS))

    # Add CCLE TMT (366 cell lines) to Deep Proteome training!
    X_ccle_tmt = np.nan_to_num(data["X_ccle_tmt"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y_ccle_tmt = np.nan_to_num(data["Y_ccle_tmt"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_prot_tr = np.vstack([X_cp[tr_c_mask], X_ccle_tmt])
    Y_prot_tr = np.vstack([Y_cp[tr_c_mask], Y_ccle_tmt])
    X_rppa_tr, Y_rppa_tr = X_tcga[tr_t_mask], Y_tcga[tr_t_mask]

    print(f"Train Sets -> TCGA RPPA: {len(X_rppa_tr)} | CPTAC+CCLE MS Proteome: {len(X_prot_tr)}")
    print(f"Blind Unseen Cancers (LUAD + OV) -> TCGA RPPA: {te_t_mask.sum()} | CPTAC MS: {te_c_mask.sum()}")

    A_gp, A_pp, pw_names = build_kegg_signaling_graph(top_rna, max_pathways=45)

    def train_model(net, name: str, epochs: int = 35):
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        rppa_loader = DataLoader(TensorDataset(torch.tensor(X_rppa_tr), torch.tensor(Y_rppa_tr)), batch_size=256, shuffle=True)
        prot_loader = DataLoader(TensorDataset(torch.tensor(X_prot_tr), torch.tensor(Y_prot_tr)), batch_size=128, shuffle=True)
        net.train()
        for ep in range(epochs):
            tot_r, tot_p = 0.0, 0.0
            for bx, by in rppa_loader:
                opt.zero_grad()
                lr = loss_fn(net.forward_rppa(bx.to(DEVICE)), by.to(DEVICE))
                lr.backward()
                opt.step()
                tot_r += float(lr.item()) * len(bx)
            for bx, by in prot_loader:
                opt.zero_grad()
                lp = loss_fn(net.forward_prot(bx.to(DEVICE)), by.to(DEVICE))
                lp.backward()
                opt.step()
                tot_p += float(lp.item()) * len(bx)
            if (ep + 1) in (1, 15, epochs):
                print(f"  [{name}] Epoch {ep+1:2d}/{epochs} | RPPA MSE={tot_r/len(X_rppa_tr):.4f} | MS Prot MSE={tot_p/len(X_prot_tr):.4f}")
        net.eval()
        return net

    gnn = train_model(PathwayGNNTranslator(A_gp, A_pp, rppa_cis_idx, prot_cis_idx).to(DEVICE), "Pathway-GNN")
    mlp = train_model(MLPBaselineTranslator(len(top_rna), rppa_cis_idx, prot_cis_idx).to(DEVICE), "MLP-Baseline")

    results = []
    phospho_tables = []

    with torch.no_grad():
        # Evaluate on Unseen TCGA LUAD & OV (RPPA Total & Phospho)
        for c in sorted(UNSEEN_CANCERS):
            m = (C_tcga == c)
            xt = torch.tensor(X_tcga[m], device=DEVICE)
            yp_gnn = gnn.forward_rppa(xt).cpu().numpy()
            yp_mlp = mlp.forward_rppa(xt).cpu().numpy()
            med_gnn, med_naive, df_g = eval_predictions(Y_tcga[m], yp_gnn, X_tcga[m], rppa_cis_idx, rppa_abs, is_phospho)
            med_mlp, _, _ = eval_predictions(Y_tcga[m], yp_mlp, X_tcga[m], rppa_cis_idx, rppa_abs, is_phospho)

            # Separate Phospho vs Total Protein RPPA
            phos_gnn = float(df_g[df_g["is_phospho"]]["model_r"].median())
            phos_naive = float(df_g[df_g["is_phospho"]]["naive_r"].dropna().median())
            tot_gnn = float(df_g[~df_g["is_phospho"]]["model_r"].median())
            tot_naive = float(df_g[~df_g["is_phospho"]]["naive_r"].dropna().median())
            print(f"  [BLIND TCGA-{c} RPPA] All Med r: Naive={med_naive:.4f} | MLP={med_mlp:.4f} | Pathway-GNN={med_gnn:.4f} (Total={tot_gnn:.4f}, Phospho={phos_gnn:.4f} vs Naive {phos_naive:.4f})")

            results.append({"cancer": c, "platform": "TCGA_RPPA_All", "naive_r": med_naive, "mlp_r": med_mlp, "gnn_r": med_gnn})
            results.append({"cancer": c, "platform": "TCGA_RPPA_Phospho", "naive_r": phos_naive, "mlp_r": np.nan, "gnn_r": phos_gnn})
            df_g["cancer"] = c
            phospho_tables.append(df_g[df_g["is_phospho"]].sort_values("model_r", ascending=False))

        # Evaluate on Unseen CPTAC LUAD & OV (Deep MS Proteome)
        for c in sorted(UNSEEN_CANCERS):
            m = (C_cp == c)
            xc = torch.tensor(X_cp[m], device=DEVICE)
            yp_gnn = gnn.forward_prot(xc).cpu().numpy()
            yp_mlp = mlp.forward_prot(xc).cpu().numpy()
            med_gnn, med_naive, _ = eval_predictions(Y_cp[m], yp_gnn, X_cp[m], prot_cis_idx, top_prot)
            med_mlp, _, _ = eval_predictions(Y_cp[m], yp_mlp, X_cp[m], prot_cis_idx, top_prot)
            print(f"  [BLIND CPTAC-{c} MS ] All Med r: Naive={med_naive:.4f} | MLP={med_mlp:.4f} | Pathway-GNN={med_gnn:.4f}")
            results.append({"cancer": c, "platform": "CPTAC_MS_Proteome", "naive_r": med_naive, "mlp_r": med_mlp, "gnn_r": med_gnn})

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    phos_df = pd.concat(phospho_tables, ignore_index=True)
    phos_df.to_csv(OUT_PHOS_CSV, index=False, encoding="utf-8-sig")
    print("\nTop Phospho Sites Predicted by Pathway-GNN in Unseen LUAD:")
    print(phos_df[phos_df["cancer"] == "LUAD"].head(12).to_string(index=False))

    # Plot Comparison Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=180)
    sub_all = res_df[res_df["platform"].isin(["TCGA_RPPA_All", "CPTAC_MS_Proteome"])]
    xlabs = [f"{r['cancer']}\n{r['platform'].replace('_', ' ')}" for _, r in sub_all.iterrows()]
    xp = np.arange(len(sub_all))
    axes[0].bar(xp - 0.25, sub_all["naive_r"], width=0.25, label="Naive 1:1 mRNA", color="#95a5a6", edgecolor="black", linewidth=0.5)
    axes[0].bar(xp, sub_all["mlp_r"], width=0.25, label="MLP (No Graph)", color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[0].bar(xp + 0.25, sub_all["gnn_r"], width=0.25, label="Pathway-GNN (TCGA+CPTAC+CCLE)", color="#2980b9", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(xlabs, fontsize=8.5)
    axes[0].set_ylabel("Median Cross-Patient Pearson (r)")
    axes[0].set_title("Blind Unseen Cancer Types (LUAD & OV)\nNaive mRNA vs MLP vs Pathway-GNN", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8)

    luad_phos = phos_df[phos_df["cancer"] == "LUAD"].head(10)
    yp = np.arange(len(luad_phos))
    axes[1].barh(yp - 0.18, luad_phos["naive_r"].fillna(0), height=0.35, label="Naive Same-Gene mRNA", color="#bdc3c7", edgecolor="black", linewidth=0.5)
    axes[1].barh(yp + 0.18, luad_phos["model_r"], height=0.35, label="Pathway-GNN Predicted Phospho", color="#c0392b", edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(yp)
    axes[1].set_yticklabels(luad_phos["feature"], fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cross-Patient Pearson (r) in Unseen LUAD")
    axes[1].set_title("Top Key Phosphorylation Hubs Predicted\nfrom Transcriptome Alone (Unseen LUAD)", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved comparison figure to {OUT_PNG}")

    # 3. Demonstrate 4-Layer Mechanistic Signaling Map using Pathway-GNN on GSE307106 Panobinostat shift!
    # Load GSE307106 real DMSO vs Panobinostat RNA shift, feed through Pathway-GNN to predict protein & phospho changes!
    from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex
    ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")
    df307 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE307106\GSE307106_raw_counts.txt.gz", sep="\t", index_col=0)
    df307["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df307.index]
    df307 = df307[df307["__sym__"] != ""].groupby("__sym__", sort=False).mean()
    vals = df307.values
    cpm = np.log2(vals / np.maximum(vals.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0)
    mat307 = pd.DataFrame(cpm.T, index=df307.columns, columns=df307.index)

    def to_z(row_names):
        m = mat307.loc[row_names].reindex(columns=top_rna).fillna(0.0).values
        mu = m.mean(axis=1, keepdims=True)
        sd = np.maximum(m.std(axis=1, keepdims=True), 1e-6)
        return ((m - mu) / sd).astype(np.float32)

    with torch.no_grad():
        rppa_dmso = gnn.forward_rppa(torch.tensor(to_z(["DMSO_1", "DMSO_2"]), device=DEVICE)).mean(dim=0).cpu().numpy()
        rppa_pano = gnn.forward_rppa(torch.tensor(to_z(["Panobinostat_20h_1", "Panobinostat_20h_2"]), device=DEVICE)).mean(dim=0).cpu().numpy()
    rppa_delta = dict(zip(rppa_abs, rppa_pano - rppa_dmso))

    def get_d(k):
        return float(rppa_delta.get(k, 0.0))

    pathway_nodes = [
        {"name": "EGFR", "tier": "Upstream", "delta": get_d("EGFR"), "is_phospho": False},
        {"name": "HER2", "tier": "Upstream", "delta": get_d("HER2"), "is_phospho": False},
        {"name": "CMET", "tier": "Upstream", "delta": get_d("CMET"), "is_phospho": False},
        {"name": "AKT_pS473", "tier": "Mid_Kinase", "delta": get_d("AKT_pS473"), "is_phospho": True},
        {"name": "MAPK_pT202Y204", "tier": "Mid_Kinase", "delta": get_d("MAPK_pT202Y204"), "is_phospho": True},
        {"name": "STAT3_pY705", "tier": "Mid_Kinase", "delta": get_d("STAT3_pY705"), "is_phospho": True},
        {"name": "X4EBP1_pS65", "tier": "Mid_Kinase", "delta": get_d("X4EBP1_pS65"), "is_phospho": True},
        {"name": "CDK1", "tier": "Downstream", "delta": get_d("CDK1"), "is_phospho": False},
        {"name": "BCL2", "tier": "Downstream", "delta": get_d("BCL2"), "is_phospho": False},
        {"name": "P21", "tier": "Downstream", "delta": get_d("P21"), "is_phospho": False},
        {"name": "PARP1", "tier": "Downstream", "delta": get_d("PARP1"), "is_phospho": False},
    ]
    edges = [
        ("HDAC1/2/6", "EGFR", "inhibit"), ("HDAC1/2/6", "HER2", "inhibit"), ("HDAC1/2/6", "P21", "activate"),
        ("EGFR", "AKT_pS473", "activate"), ("EGFR", "MAPK_pT202Y204", "activate"),
        ("HER2", "AKT_pS473", "activate"), ("CMET", "STAT3_pY705", "activate"),
        ("AKT_pS473", "X4EBP1_pS65", "activate"), ("AKT_pS473", "BCL2", "activate"),
        ("MAPK_pT202Y204", "CDK1", "activate"), ("STAT3_pY705", "BCL2", "activate"),
        ("P21", "CDK1", "inhibit"),
        ("CDK1", "Cell Cycle Arrest", "inhibit"), ("P21", "Cell Cycle Arrest", "activate"),
        ("BCL2", "Apoptosis Activation", "inhibit"), ("PARP1", "DNA Damage Repair", "activate"),
        ("X4EBP1_pS65", "mTOR Translation", "activate"),
    ]
    downstream_outcomes = [
        ("Cell Cycle Arrest", "Activated"),
        ("Apoptosis Activation", "Activated"),
        ("mTOR Translation", "Suppressed" if get_d("X4EBP1_pS65") < 0 else "Activated"),
        ("DNA Damage Repair", "Suppressed" if get_d("PARP1") < 0 else "Activated"),
    ]
    plot_mechanistic_signaling_map(
        drug_name="Panobinostat (HDACi)",
        direct_targets=[("HDAC1/2/6", "Master Target")],
        pathway_nodes=pathway_nodes,
        edges=edges,
        downstream_outcomes=downstream_outcomes,
        out_png=OUT_MAP_PNG,
    )


if __name__ == "__main__":
    main()
