# -*- coding: utf-8 -*-
"""
exp42_lincs_l1000_mega_scaling_law.py
Exp42: Mega-Scale LINCS L1000 (GSE70138, 118,050 signatures -> 1,500+ unique small-molecule compounds)
       Strict Leave-Drug-Out Zero-Shot Benchmark & Multi-Thousand-Compound Data Scaling Law
       (50 -> 150 -> 300 -> 600 -> 1,000+ training compounds -> ~300+ completely unseen drugs).
"""

import os
import sys
import gzip
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from chem_interface import encode_smiles_multimodal

SEED = 42
np.random.seed(SEED)

LINCS_DIR = r"D:\data\AIVC\data\lincs_l1000"
GCTX_GZ = os.path.join(LINCS_DIR, "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz")
GCTX_RAW = os.path.join(LINCS_DIR, "GSE70138_Level5.gctx")
PERT_INFO = os.path.join(LINCS_DIR, "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz")
SIG_INFO = os.path.join(LINCS_DIR, "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz")
GENE_INFO = os.path.join(LINCS_DIR, "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz")
CHEMBERTA_DIR = r"D:\data\AIVC\pretrained\chemberta"
LINCS_CACHE = os.path.join(LINCS_DIR, "gse70138_consensus_lm978_cache.npz")

OUT_CSV = r"D:\data\AIVC\test_results\exp42_lincs_l1000_mega_scaling_results.csv"


def eval_drug_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p_all, p_top20, d_top20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p_all.append(float(pearsonr(yt, yp)[0]))
        top_idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[top_idx]) > 1e-6 and np.std(yp[top_idx]) > 1e-6:
            p_top20.append(float(pearsonr(yt[top_idx], yp[top_idx])[0]))
        d_top20.append(float((np.sign(yt[top_idx]) == np.sign(yp[top_idx])).mean()))
    return {
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def load_or_build_lincs_consensus():
    if os.path.exists(LINCS_CACHE):
        ck = np.load(LINCS_CACHE, allow_pickle=True)
        print(f"  [LINCS Cache] Loaded {len(ck['pert_ids'])} compounds from {LINCS_CACHE}")
        return [str(x) for x in ck["pert_ids"]], ck["Y_all"].astype(np.float32), ck["X_mol_all"].astype(np.float32)

    if not os.path.exists(GCTX_RAW):
        print(f"  [Decompress] Streaming {os.path.basename(GCTX_GZ)} -> {os.path.basename(GCTX_RAW)} ...")
        t_d = time.time()
        with gzip.open(GCTX_GZ, "rb") as f_in, open(GCTX_RAW, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
        print(f"  [Decompress] Done in {time.time()-t_d:.1f}s")

    p_df = pd.read_csv(PERT_INFO, sep="\t", low_memory=False)
    p_df = p_df[(p_df["pert_type"] == "trt_cp") & p_df["canonical_smiles"].notna()]
    p_df = p_df[~p_df["canonical_smiles"].astype(str).isin(["-666", "restricted", ""])]
    pid_to_smi = dict(zip(p_df["pert_id"].astype(str), p_df["canonical_smiles"].astype(str)))

    s_df = pd.read_csv(SIG_INFO, sep="\t", low_memory=False)
    s_df = s_df[(s_df["pert_type"] == "trt_cp") & (s_df["pert_id"].astype(str).isin(pid_to_smi))]
    sig_to_pid = dict(zip(s_df["sig_id"].astype(str), s_df["pert_id"].astype(str)))

    g_df = pd.read_csv(GENE_INFO, sep="\t", low_memory=False)
    lm_ids = set(g_df[g_df["pr_is_lm"] == 1]["pr_gene_id"].astype(str).tolist())

    with h5py.File(GCTX_RAW, "r") as hf:
        mat_ds = hf["0/DATA/0/matrix"]  # shape: (n_sigs, n_genes) in h5py C-order!
        row_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in hf["0/META/ROW/id"][:]]
        col_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in hf["0/META/COL/id"][:]]
        print(f"  [GCTX] matrix shape={mat_ds.shape}, len(row_ids)={len(row_ids)}, len(col_ids)={len(col_ids)}")

        # Determine which axis is genes (12328) and which is signatures (118050)
        if mat_ds.shape[1] == len(row_ids):
            # mat_ds is (n_sigs, n_genes), col_ids = sigs, row_ids = genes
            gene_idx = np.array([i for i, gid in enumerate(row_ids) if gid in lm_ids], dtype=np.int64)
            sig_list = col_ids
            sig_axis = 0
        else:
            gene_idx = np.array([i for i, gid in enumerate(col_ids) if gid in lm_ids], dtype=np.int64)
            sig_list = row_ids
            sig_axis = 1

        # Pick up to 6 signatures per compound to keep read fast and balanced
        pid_sig_indices = {}
        for i, sid in enumerate(sig_list):
            pid = sig_to_pid.get(sid)
            if pid is not None:
                lst = pid_sig_indices.setdefault(pid, [])
                if len(lst) < 6:
                    lst.append(i)

        valid_pids = sorted(pid_sig_indices.keys())
        all_sig_rows = sorted({idx for lst in pid_sig_indices.values() for idx in lst})
        idx_to_pid = {idx: pid for pid, lst in pid_sig_indices.items() for idx in lst}
        print(f"  [LINCS L1000] Aggregating {len(all_sig_rows)} signatures across {len(valid_pids)} unique compounds (978 landmark genes)...")

        sums = {pid: np.zeros(len(gene_idx), dtype=np.float64) for pid in valid_pids}
        cnts = {pid: 0 for pid in valid_pids}
        chunk = 2000
        for s0 in range(0, len(all_sig_rows), chunk):
            sub = all_sig_rows[s0:s0 + chunk]
            if sig_axis == 0:
                block = mat_ds[sub, :][:, gene_idx].astype(np.float32)
            else:
                block = mat_ds[:, sub][gene_idx, :].T.astype(np.float32)
            for li, r_idx in enumerate(sub):
                pid = idx_to_pid[r_idx]
                sums[pid] += np.clip(block[li], -10.0, 10.0)
                cnts[pid] += 1

    final_pids = [p for p in valid_pids if cnts[p] > 0]
    Y_all = np.vstack([(sums[p] / cnts[p]).astype(np.float32) for p in final_pids])
    print(f"  [LINCS L1000] Encoding {len(final_pids)} compound SMILES with ChemBERTa + Morgan...")
    morg, cb = encode_smiles_multimodal([pid_to_smi[p] for p in final_pids], chemberta_dir=CHEMBERTA_DIR, device_str="cpu")
    X_mol_all = np.hstack([cb, morg]).astype(np.float32)
    np.savez_compressed(LINCS_CACHE, pert_ids=np.array(final_pids), Y_all=Y_all, X_mol_all=X_mol_all)
    return final_pids, Y_all, X_mol_all


def main():
    t0 = time.time()
    print("=== Exp42: LINCS L1000 (GSE70138) Multi-Thousand-Compound Leave-Drug-Out Scaling Law ===")
    pids, Y_all, X_mol_all = load_or_build_lincs_consensus()
    n_drugs = len(pids)
    print(f"  Total unique valid small-molecule compounds: {n_drugs} x {Y_all.shape[1]} landmark genes")

    # Multi-view chemical kernels: ChemBERTa semantic cosine + Morgan structural Tanimoto/cosine + RBF
    X_cb = normalize(X_mol_all[:, :384], norm="l2")
    X_mg = normalize(X_mol_all[:, 384:], norm="l2")
    K_cb = (X_cb @ X_cb.T).astype(np.float64)
    K_mg = (X_mg @ X_mg.T).astype(np.float64)
    K_rbf = np.exp(-1.5 * np.clip(1.0 - K_mg, 0.0, 2.0))

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_drugs)
    n_tr_total = int(0.8 * n_drugs)
    tr_full = perm[:n_tr_total]
    ev_idx = perm[n_tr_total:]
    Y_ev = Y_all[ev_idx]
    print(f"  Strict Leave-Drug-Out Split: Train={len(tr_full)} compounds -> Completely Unseen Test={len(ev_idx)} compounds")

    rows = []
    # Baselines at 100%
    Y_tr_100 = Y_all[tr_full]
    p_mean = np.repeat(Y_tr_100.mean(axis=0, keepdims=True), len(ev_idx), axis=0).astype(np.float32)
    p_ridge = Ridge(alpha=300.0).fit(X_mol_all[tr_full], Y_tr_100).predict(X_mol_all[ev_idx]).astype(np.float32)

    for mname, pmat in [("Train_Mean", p_mean), ("Linear_Ridge_SMILES", p_ridge)]:
        m = eval_drug_metrics(Y_ev, pmat)
        print(f"  [{mname:28s} | n={len(tr_full):4d}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.4f}")
        rows.append({"model": mname, "n_train_drugs": len(tr_full), "n_test_drugs": len(ev_idx), **m})

    # Mega Scaling Law across 50 -> 150 -> 300 -> 600 -> 1000 -> 100% compounds
    steps = [s for s in [50, 150, 300, 600, 1000] if s < len(tr_full)] + [len(tr_full)]
    for n_sub in steps:
        sub = tr_full[:n_sub]
        K_tr = 0.25 * K_cb[np.ix_(sub, sub)] + 0.35 * K_mg[np.ix_(sub, sub)] + 0.40 * K_rbf[np.ix_(sub, sub)]
        K_ev = 0.25 * K_cb[np.ix_(ev_idx, sub)] + 0.35 * K_mg[np.ix_(ev_idx, sub)] + 0.40 * K_rbf[np.ix_(ev_idx, sub)]
        mu = Y_all[sub].mean(axis=0, keepdims=True)
        alpha = np.linalg.solve(K_tr + 1.8 * np.eye(n_sub, dtype=np.float64), (Y_all[sub] - mu).astype(np.float64))
        p_mk = (K_ev @ alpha + mu).astype(np.float32)
        m = eval_drug_metrics(Y_ev, p_mk)
        print(f"  [AIVC_MultiKernel_Chem       | n={n_sub:4d}] MeanR={m['mean_pearson']:.4f} | Top20R={m['top20_pearson']:.4f} | DirAcc={m['dir_acc_top20']:.4f} | MSE={m['mse']:.4f}")
        rows.append({"model": "AIVC_MultiKernel_Chem", "n_train_drugs": n_sub, "n_test_drugs": len(ev_idx), **m})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved Exp42 LINCS L1000 Mega Scaling Law -> {OUT_CSV} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
