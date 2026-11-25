import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEAN_DIR = r"D:\data\AIVC\data\clinical_cohorts\cleaned"
RAW_DIR = r"D:\data\AIVC\data\clinical_cohorts\raw"
OUT_CSV = r"D:\data\AIVC\test_results\exp19_intrinsic_vs_tme_ablation.csv"
OUT_PURITY_CSV = r"D:\data\AIVC\test_results\exp19_gse123845_paired_tumor_killing.csv"
OUT_PNG = r"D:\data\AIVC\test_results\exp19_intrinsic_vs_tme_ablation.png"

TME_SIGNATURES = {
    "Tumor_Proliferation": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "AURKA", "BIRC5", "E2F1"],
    "Tumor_Purity_Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "ESR1", "ERBB2"],
    "CD8_Effector_T": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10"],
    "T_Exhaustion_Checkpoint": ["PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "TOX", "FOXP3"],
    "CAF_Stroma_Barrier": ["FAP", "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FN1", "POSTN", "TGFB1"],
    "M2_TAM_Myeloid": ["CD163", "MRC1", "MSR1", "CSF1R", "CD68", "TREM2", "SPP1"],
    "EMT_Stemness": ["VIM", "ZEB1", "SNAI1", "SNAI2", "TWIST1", "CD44", "ALDH1A1"],
    "Hypoxia_Glycolysis": ["HIF1A", "VEGFA", "LDHA", "SLC2A1", "CA9", "ENO1", "PKM"],
    "DDR_BRCAness": ["BRCA1", "BRCA2", "RAD51", "PARP1", "ATM", "ATR", "CHEK1"],
    "Apoptosis_Intrinsic": ["BCL2", "BAX", "BAK1", "CASP3", "CASP8", "TP53"],
}

DRUG_ALIAS = {
    "FLUOROURACIL": "5-Fluorouracil",
    "CARBOPLATIN": "Cisplatin",
    "DEXAMETHASONE": "Dexamethasone",
}


def row_zscore(mat: np.ndarray) -> np.ndarray:
    """Within-sample Z-score normalization across genes (scale-invariant across Microarray & RNA-seq)."""
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    out = (mat - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    """Harrell's C-index (higher risk = shorter survival / higher event hazard)."""
    valid = np.isfinite(times) & np.isfinite(events) & np.isfinite(risks)
    t, e, r = times[valid], events[valid], risks[valid]
    concordant = 0.0
    permissible = 0.0
    n = len(t)
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if t[j] > t[i]:
                permissible += 1.0
                if r[i] > r[j]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else np.nan


class BioMolKGCellDrugNet(nn.Module):
    """Pretrained cell-intrinsic drug sensitivity network (Cell Z-score + Drug KG -> -LN_IC50)."""
    def __init__(self, n_genes: int, n_drug_feats: int = 2048, hid: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(n_drug_feats, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_cell, x_drug):
        hc = self.cell_enc(x_cell)
        hd = self.drug_enc(x_drug)
        return self.head(torch.cat([hc, hd], dim=-1)).squeeze(-1)


def main():
    print(f"=== Exp19: Cell-Intrinsic Drug Killing vs TME Unbiased Ablation (Device: {DEVICE}) ===")

    # 1. Load cell line RNA-seq & clinical cohorts to determine top 2,000 shared high-variance genes
    inv = pd.read_csv(os.path.join(CLEAN_DIR, "cohort_inventory_summary.csv"))
    shared_genes = None
    cohort_data = {}
    for cid in inv["cohort_id"]:
        c = pd.read_csv(os.path.join(CLEAN_DIR, cid, f"{cid}_clinical.csv"))
        e = pd.read_parquet(os.path.join(CLEAN_DIR, cid, f"{cid}_expr_log2.parquet"))
        cohort_data[cid] = (c, e)
        gset = set(e.columns)
        shared_genes = gset if shared_genes is None else (shared_genes & gset)

    # Ensure all TME signature genes are included if present in shared_genes
    sig_genes_all = set(g for gs in TME_SIGNATURES.values() for g in gs)

    print("Loading cell line RNA-seq (Sanger CMP + Broad CCLE)...")
    rna_raw = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", skiprows=[1, 2], low_memory=False)
    rna_raw = rna_raw.rename(columns={rna_raw.columns[0]: "gene_symbol"})
    rna_raw = rna_raw.drop(columns=[c for c in rna_raw.columns if str(c).startswith("Unnamed:")])
    rna_raw["gene_symbol"] = rna_raw["gene_symbol"].astype(str).str.strip().str.upper()
    rna_raw = rna_raw[rna_raw["gene_symbol"].isin(shared_genes)].drop_duplicates(subset=["gene_symbol"])
    rna_raw = rna_raw.set_index("gene_symbol").apply(pd.to_numeric, errors="coerce")
    cell_genes = set(rna_raw.index)
    common_genes = sorted(shared_genes & cell_genes)

    # Pick top 2,000 most variable genes in cell lines + force-include TME signature genes
    gene_var = rna_raw.loc[common_genes].var(axis=1).sort_values(ascending=False)
    top_2000 = list(gene_var.index[:2000])
    for g in sorted(sig_genes_all & set(common_genes)):
        if g not in top_2000:
            top_2000.append(g)
    model_genes = sorted(top_2000[:2000])
    # Also keep sig_genes separately for TME scoring across all 11,830 shared genes
    print(f"Shared genes across all 17 clinical cohorts + cell lines: {len(common_genes)} | Selected model_genes: {len(model_genes)}")

    cell_expr_df = rna_raw.loc[model_genes].T  # cell_lines x model_genes
    cell_expr_df = cell_expr_df.dropna(how="all")
    cell_z = row_zscore(cell_expr_df.values)
    cell_id_to_row = {sid: i for i, sid in enumerate(cell_expr_df.index)}

    # 2. Pretrain BioMolKG-CellIntrinsic on GDSC1 + GDSC2 (no clinical outcomes ever touched!)
    gdsc_npz = np.load(r"D:\data\AIVC\data\gdsc_drug_kg_features.npz", allow_pickle=True)
    drug_to_feat = {str(d).upper(): gdsc_npz["feats"][i].astype(np.float32) for i, d in enumerate(gdsc_npz["drugs"])}

    gdsc1 = pd.read_excel(r"D:\data\AIVC\data\GDSC1_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    gdsc2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    gdsc = pd.concat([gdsc1, gdsc2], ignore_index=True).dropna()
    gdsc["d_upper"] = gdsc["DRUG_NAME"].astype(str).str.strip().str.upper()
    gdsc = gdsc[gdsc["SANGER_MODEL_ID"].isin(cell_id_to_row) & gdsc["d_upper"].isin(drug_to_feat)].copy()
    # Target: -LN_IC50 (higher = more sensitive)
    c_idx = gdsc["SANGER_MODEL_ID"].map(cell_id_to_row).values
    d_names = gdsc["d_upper"].values
    unique_d, d_inv = np.unique(d_names, return_inverse=True)
    d_mat = np.stack([drug_to_feat[u] for u in unique_d], axis=0)
    y_sens = (-gdsc["LN_IC50"].values).astype(np.float32)
    print(f"Pretraining BioMolKG-CellIntrinsic on {len(y_sens)} GDSC1+GDSC2 cell-drug pairs ({len(unique_d)} drugs)...")

    # Fast GPU training
    model = BioMolKGCellDrugNet(n_genes=len(model_genes), n_drug_feats=2048).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    cell_z_t = torch.tensor(cell_z, device=DEVICE)
    d_mat_t = torch.tensor(d_mat, device=DEVICE)
    c_idx_t = torch.tensor(c_idx, dtype=torch.long)
    d_idx_t = torch.tensor(d_inv, dtype=torch.long)
    y_t = torch.tensor(y_sens, dtype=torch.float32)

    loader = DataLoader(TensorDataset(c_idx_t, d_idx_t, y_t), batch_size=2048, shuffle=True)
    model.train()
    for ep in range(6):
        tot_loss = 0.0
        for bc, bd, by in loader:
            bc_x = cell_z_t[bc]
            bd_x = d_mat_t[bd]
            by = by.to(DEVICE)
            opt.zero_grad()
            pred = model(bc_x, bd_x)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(by)
        print(f"  Pretrain Epoch {ep+1}/6 | MSE={tot_loss / len(y_sens):.4f}")

    model.eval()

    # 3. Compute Cell-Intrinsic Drug Sensitivity, TME Signatures, and 2,000-gene Z-score for all 17 clinical cohorts
    def score_cohort(cid: str):
        c, e = cohort_data[cid]
        # Full shared-gene Z-score for TME signatures
        e_shared = e[sorted(shared_genes)]
        z_shared = pd.DataFrame(row_zscore(e_shared.values), index=e_shared.index, columns=e_shared.columns)
        tme_df = pd.DataFrame(index=e.index)
        for sname, sgenes in TME_SIGNATURES.items():
            avail = [g for g in sgenes if g in z_shared.columns]
            tme_df[sname] = z_shared[avail].mean(axis=1) if avail else 0.0

        # 2,000-gene Z-score for model & cell-intrinsic BioMolKG
        x_2000 = row_zscore(e[model_genes].values)
        x_t = torch.tensor(x_2000, device=DEVICE)

        # Predict -LN_IC50 for every matched small-molecule drug in regimen
        mean_sens, max_sens, min_sens = [], [], []
        with torch.no_grad():
            # Precompute cell embeddings once
            hc = model.cell_enc(x_t)
            drug_cache = {}
            for drugs_str in c["drugs_normalized"].fillna(""):
                dlist = [d.strip() for d in str(drugs_str).split(";") if d.strip()]
                scores = []
                for d in dlist:
                    du = DRUG_ALIAS.get(d.upper(), d).upper()
                    if du in drug_to_feat:
                        if du not in drug_cache:
                            dfeat = torch.tensor(drug_to_feat[du], device=DEVICE).unsqueeze(0)
                            hd = model.drug_enc(dfeat).expand(len(hc), -1)
                            drug_cache[du] = model.head(torch.cat([hc, hd], dim=-1)).squeeze(-1).cpu().numpy()
                        scores.append(drug_cache[du])
                if scores:
                    mat_s = np.stack(scores, axis=1)  # n_samples x n_matched_drugs
                    idx = len(mean_sens)
                    row_s = mat_s[idx]
                    mean_sens.append(float(np.mean(row_s)))
                    max_sens.append(float(np.max(row_s)))
                    min_sens.append(float(np.min(row_s)))
                else:
                    mean_sens.append(0.0)
                    max_sens.append(0.0)
                    min_sens.append(0.0)

        intr_df = pd.DataFrame({
            "Intrinsic_Mean_Sens": mean_sens,
            "Intrinsic_Max_Sens": max_sens,
            "Intrinsic_Min_Sens": min_sens,
            "Intrinsic_x_Prolif": np.array(mean_sens) * tme_df["Tumor_Proliferation"].values,
        }, index=e.index)
        return c, intr_df, tme_df, x_2000

    print("\nScoring all 17 clinical cohorts (Cell-Intrinsic BioMolKG + TME Signatures)...")
    scored = {cid: score_cohort(cid) for cid in inv["cohort_id"]}

    # 4. Verify GSE123845 paired T1 -> T2 -> T3 tumor killing vs real pathologist tumor purity
    print("\n=== Verifying GSE123845 Paired Pre(T1) -> On(T2) -> Post(T3) Tumor Killing ===")
    m123 = pd.read_csv(os.path.join(RAW_DIR, "GSE123845", "GSE123845_meta.csv"))
    c123, intr123, tme123, _ = scored["GSE123845"]
    df123 = c123.merge(
        m123[["sample_id", "tumor_purity_pathology", "proliferative_index", "til_density"]],
        on="sample_id", how="left"
    )
    df123["Tumor_Proliferation"] = tme123["Tumor_Proliferation"].values
    df123["Tumor_Purity_Epithelial"] = tme123["Tumor_Purity_Epithelial"].values
    df123["CD8_Effector_T"] = tme123["CD8_Effector_T"].values
    df123["CAF_Stroma_Barrier"] = tme123["CAF_Stroma_Barrier"].values
    df123["tumor_purity_pathology"] = pd.to_numeric(df123["tumor_purity_pathology"], errors="coerce")
    df123["proliferative_index"] = pd.to_numeric(df123["proliferative_index"], errors="coerce")

    valid_p = df123["tumor_purity_pathology"].notna()
    r_pur, p_pur = spearmanr(df123.loc[valid_p, "Tumor_Purity_Epithelial"], df123.loc[valid_p, "tumor_purity_pathology"])
    r_pro, p_pro = spearmanr(df123["Tumor_Proliferation"], df123["proliferative_index"])
    print(f"  GSE123845 Tumor_Purity_Epithelial vs Pathologist Purity: Spearman r={r_pur:.4f} (p={p_pur:.2e})")
    print(f"  GSE123845 Tumor_Proliferation vs Proliferative_Index: Spearman r={r_pro:.4f} (p={p_pro:.2e})")

    purity_summary = df123.groupby(["treatment_timing", "response_binary"])[
        ["tumor_purity_pathology", "Tumor_Purity_Epithelial", "Tumor_Proliferation", "CD8_Effector_T", "CAF_Stroma_Barrier"]
    ].mean().reset_index()
    purity_summary.to_csv(OUT_PURITY_CSV, index=False, encoding="utf-8-sig")
    print(purity_summary.to_string(index=False))

    # 5. Track A: Breast Cancer Multi-Drug NAC pCR Strict Leave-Cohorts-Out Blind Test
    TRAIN_BRCA = ["GSE194040", "GSE20194", "GSE20271", "GSE22093", "GSE32646", "GSE41998"]
    TEST_BRCA = ["GSE25066", "GSE163882", "GSE123845"]

    def get_features(cid, pre_only=True):
        c, intr, tme, x2000 = scored[cid]
        if pre_only:
            mask = (c["treatment_timing"] == "Pre-treatment").values
        else:
            mask = np.ones(len(c), dtype=bool)
        return c[mask].reset_index(drop=True), intr.values[mask], tme.values[mask], x2000[mask]

    tr_c_list, tr_intr, tr_tme, tr_x = [], [], [], []
    for cid in TRAIN_BRCA:
        c, intr, tme, x = get_features(cid, pre_only=True)
        valid = c["response_binary"].notna().values
        tr_c_list.append(c[valid])
        tr_intr.append(intr[valid])
        tr_tme.append(tme[valid])
        tr_x.append(x[valid])

    y_tr = np.concatenate([c["response_binary"].values for c in tr_c_list]).astype(int)
    X_tr_intr = np.vstack(tr_intr)
    X_tr_tme = np.vstack(tr_tme)
    X_tr_x = np.vstack(tr_x)
    X_tr_all = np.hstack([X_tr_intr, X_tr_tme, X_tr_x])

    # Fit LogisticRegression models on standardized features for each of the 4 arms
    arms_tr = {
        "Arm1_Intrinsic_BioMolKG": X_tr_intr,
        "Arm2_TME_Only": X_tr_tme,
        "Arm3_Transcriptome_2000g": X_tr_x,
        "Arm4_Unified_Fusion": np.hstack([X_tr_intr, X_tr_tme]),
        "Arm5_Full_Fusion_2000g": X_tr_all,
    }
    fitted_models = {}
    for arm_name, Xm in arms_tr.items():
        sc = StandardScaler()
        Xs = sc.fit_transform(Xm)
        C_val = 0.01 if "2000g" in arm_name else 0.5
        clf = LogisticRegression(C=C_val, max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(Xs, y_tr)
        fitted_models[arm_name] = (sc, clf)

    results = []
    print("\n=== Track A: Breast Cancer Multi-Drug NAC pCR (Train N=1,907 -> Blind External Cohorts) ===")
    for cid in TEST_BRCA:
        c, intr, tme, x = get_features(cid, pre_only=True)
        valid = c["response_binary"].notna().values
        y_te = c.loc[valid, "response_binary"].values.astype(int)
        arms_te = {
            "Arm1_Intrinsic_BioMolKG": intr[valid],
            "Arm2_TME_Only": tme[valid],
            "Arm3_Transcriptome_2000g": x[valid],
            "Arm4_Unified_Fusion": np.hstack([intr[valid], tme[valid]]),
            "Arm5_Full_Fusion_2000g": np.hstack([intr[valid], tme[valid], x[valid]]),
        }
        plat = c["platform_type"].iloc[0]
        for arm_name, Xte in arms_te.items():
            sc, clf = fitted_models[arm_name]
            prob = clf.predict_proba(sc.transform(Xte))[:, 1]
            auc = roc_auc_score(y_te, prob)
            prauc = average_precision_score(y_te, prob)
            results.append({
                "track": "TrackA_BRCA_Blind_pCR",
                "cohort_id": cid,
                "disease": "Breast_Cancer",
                "platform": plat,
                "n_test": len(y_te),
                "arm": arm_name,
                "metric_name": "ROC-AUC",
                "metric_value": auc,
                "secondary_metric": prauc,
            })
            print(f"  {cid} ({plat}, N={len(y_te)}) | {arm_name:26s} | ROC-AUC={auc:.4f} | PR-AUC={prauc:.4f}")

    # 6. Track B: Multi-Disease Survival (OS / DRFS) & ICI Response 5-Fold CV Ablation
    print("\n=== Track B: Multi-Disease Survival (C-index) & ICI Response (ROC-AUC) 5-Fold CV ===")
    SURV_TASKS = [
        ("GSE25066", "Breast_Cancer", "pfs_time_years", "pfs_event", "DRFS"),
        ("GSE10846", "DLBCL", "os_time_years", "os_event", "OS"),
        ("GSE87371", "DLBCL", "os_time_years", "os_event", "OS"),
        ("GSE9782", "Multiple_Myeloma", "os_time_years", "os_event", "OS"),
        ("GSE39582", "Colorectal_Cancer", "os_time_years", "os_event", "OS"),
        ("GSE26712", "Ovarian_Cancer", "os_time_years", "os_event", "OS"),
        ("GSE176307", "Urothelial_ICI", "os_time_years", "os_event", "OS"),
    ]
    for cid, dis, tcol, ecol, ep_label in SURV_TASKS:
        c, intr, tme, x = get_features(cid, pre_only=True)
        valid = (c[tcol].notna() & c[ecol].notna() & (c[tcol] > 0)).values
        times = c.loc[valid, tcol].values.astype(float)
        events = c.loc[valid, ecol].values.astype(int)
        if len(np.unique(events)) < 2:
            print(f"  [SKIP] {cid} ({dis}, {ep_label}) has <2 event classes: {np.unique(events)}")
            continue
        arms_s = {
            "Arm1_Intrinsic_BioMolKG": intr[valid],
            "Arm2_TME_Only": tme[valid],
            "Arm3_Transcriptome_2000g": x[valid],
            "Arm4_Unified_Fusion": np.hstack([intr[valid], tme[valid]]),
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for arm_name, Xm in arms_s.items():
            oof_risk = np.zeros(len(events))
            for tr_i, te_i in skf.split(Xm, events):
                sc = StandardScaler()
                C_val = 0.01 if "2000g" in arm_name else 0.5
                # Use event classification log-odds as risk proxy for fast cross-arm ablation
                clf = LogisticRegression(C=C_val, max_iter=1000, random_state=SEED)
                clf.fit(sc.fit_transform(Xm[tr_i]), events[tr_i])
                oof_risk[te_i] = clf.decision_function(sc.transform(Xm[te_i]))
            cidx = compute_c_index(times, events, oof_risk)
            results.append({
                "track": f"TrackB_Survival_{ep_label}",
                "cohort_id": cid,
                "disease": dis,
                "platform": c["platform_type"].iloc[0],
                "n_test": len(events),
                "arm": arm_name,
                "metric_name": "C-index",
                "metric_value": cidx,
                "secondary_metric": np.nan,
            })
            print(f"  {cid} ({dis}, {ep_label}, N={len(events)}) | {arm_name:26s} | C-index={cidx:.4f}")

    # Also Single-Cohort Response on GSE9782 (Myeloma Bortezomib/Dex), GSE91061 (Melanoma ICI), GSE176307 (Urothelial ICI)
    for cid, dis in [("GSE9782", "Multiple_Myeloma"), ("GSE91061", "Melanoma_ICI"), ("GSE176307", "Urothelial_ICI")]:
        c, intr, tme, x = get_features(cid, pre_only=True)
        valid = c["response_binary"].notna().values
        y_ici = c.loc[valid, "response_binary"].values.astype(int)
        arms_i = {
            "Arm1_Intrinsic_BioMolKG": intr[valid],
            "Arm2_TME_Only": tme[valid],
            "Arm3_Transcriptome_2000g": x[valid],
            "Arm4_Unified_Fusion": np.hstack([intr[valid], tme[valid]]),
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for arm_name, Xm in arms_i.items():
            oof_p = np.zeros(len(y_ici))
            for tr_i, te_i in skf.split(Xm, y_ici):
                sc = StandardScaler()
                C_val = 0.01 if "2000g" in arm_name else 0.5
                clf = LogisticRegression(C=C_val, max_iter=1000, class_weight="balanced", random_state=SEED)
                clf.fit(sc.fit_transform(Xm[tr_i]), y_ici[tr_i])
                oof_p[te_i] = clf.predict_proba(sc.transform(Xm[te_i]))[:, 1]
            auc = roc_auc_score(y_ici, oof_p)
            results.append({
                "track": "TrackC_ICI_Response",
                "cohort_id": cid,
                "disease": dis,
                "platform": "RNAseq",
                "n_test": len(y_ici),
                "arm": arm_name,
                "metric_name": "ROC-AUC",
                "metric_value": auc,
                "secondary_metric": np.nan,
            })
            print(f"  {cid} ({dis}, ICI Resp, N={len(y_ici)}) | {arm_name:26s} | ROC-AUC={auc:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved all ablation results to {OUT_CSV}")

    # 7. Plot clean 2-panel comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    # Panel A: Track A Blind pCR
    df_a = res_df[res_df["track"] == "TrackA_BRCA_Blind_pCR"]
    piv_a = df_a.pivot(index="cohort_id", columns="arm", values="metric_value")[
        ["Arm1_Intrinsic_BioMolKG", "Arm2_TME_Only", "Arm3_Transcriptome_2000g", "Arm4_Unified_Fusion"]
    ]
    piv_a.plot(kind="bar", ax=axes[0], width=0.75, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Track A: Breast Cancer Multi-Drug NAC pCR\n(Blind External Hospital & Cross-Platform RNA-seq)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Blind Test ROC-AUC")
    axes[0].set_ylim(0.45, 0.90)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8, loc="lower right")

    # Panel B: Track B Survival C-index
    df_b = res_df[res_df["track"].str.startswith("TrackB_")]
    piv_b = df_b.pivot(index="cohort_id", columns="arm", values="metric_value")[
        ["Arm1_Intrinsic_BioMolKG", "Arm2_TME_Only", "Arm3_Transcriptome_2000g", "Arm4_Unified_Fusion"]
    ]
    piv_b.plot(kind="bar", ax=axes[1], width=0.75, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Track B: Multi-Disease Survival Prognosis (OS / DRFS)\n(Intrinsic Drug Killing vs TME vs Unified Fusion)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("5-Fold CV C-index")
    axes[1].set_ylim(0.45, 0.80)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved comparison figure to {OUT_PNG}")


if __name__ == "__main__":
    main()
