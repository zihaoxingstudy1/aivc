# -*- coding: utf-8 -*-
"""
sota_bench_part2_aivc_and_fm.py
Run AIVC v1.1 (Exp31), AIVC v1.1 5-Member Ensemble (Exp34),
Geneformer-V2-104M (frozen foundation backbone + perturbation adapter),
and scFoundation-100M (gene embedding + perturbation adapter)
on the exact frozen rpe1_ood_bundle.npz (7-Lineage -> Unseen RPE1, 2,016 perturbations).
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp31_lineage_invariant_gnn_proteome import (
    LineageInvariantPathwayNet, predict_lineage_prot_phos_via_exp28
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
GF_DIR = r"D:\data\AIVC\pretrained\geneformer"
SCF_DIR = r"D:\data\AIVC\pretrained\scfoundation"


def run_geneformer_v2(bundle: dict, d_rpe1: dict) -> np.ndarray:
    print("\n--- [SOTA FM 1/2] Geneformer-V2-104M_CLcancer ---")
    with open(os.path.join(GF_DIR, "geneformer", "gene_name_id_dict_gc104M.pkl"), "rb") as f:
        gene_name_to_ens = pickle.load(f)
    with open(os.path.join(GF_DIR, "geneformer", "token_dictionary_gc104M.pkl"), "rb") as f:
        ens_to_tok = pickle.load(f)

    gf_model_dir = os.path.join(GF_DIR, "Geneformer-V2-104M_CLcancer")
    gf = BertModel.from_pretrained(gf_model_dir).to(DEVICE)
    gf.eval()
    tok_emb_weight = gf.embeddings.word_embeddings.weight.detach().cpu().numpy()  # (20275, 768)

    target_genes = [str(g) for g in bundle["target_genes"]]
    ctrl_names = [str(c) for c in bundle["ctrl_names"]]
    ctrl_vecs = bundle["ctrl_vecs"]

    # Map gene symbols -> Geneformer token IDs
    sym_to_tok = {}
    for g in set(target_genes) | set(bundle["tr_perts"].tolist()) | set(bundle["ev_keys"].tolist()):
        gu = str(g).upper()
        ens = gene_name_to_ens.get(g) or gene_name_to_ens.get(gu)
        if ens is not None and ens in ens_to_tok:
            sym_to_tok[gu] = int(ens_to_tok[ens])
    print(f"  Geneformer-V2 mapped {len(sym_to_tok)} genes to token vocab")

    # Rank-value encode each lineage control vector -> 768-dim cell context embedding
    cls_id = ens_to_tok.get("<cls>", 1)
    cell_embs = {}
    with torch.no_grad():
        for cname, cvec in zip(ctrl_names, ctrl_vecs):
            # Rank non-zero genes descending by expression
            valid = [(i, float(cvec[i])) for i in range(len(target_genes)) if target_genes[i].upper() in sym_to_tok and cvec[i] > 0]
            valid.sort(key=lambda x: -x[1])
            toks = [cls_id] + [sym_to_tok[target_genes[i].upper()] for i, _ in valid[:1500]]
            inp = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
            out = gf(input_ids=inp).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
            cell_embs[cname] = out
    del gf
    torch.cuda.empty_cache()

    def get_pert_emb(p):
        tid = sym_to_tok.get(str(p).upper(), -1)
        return tok_emb_weight[tid].astype(np.float32) if tid >= 0 else np.zeros(tok_emb_weight.shape[1], dtype=np.float32)

    X_tr = np.hstack([
        np.vstack([cell_embs[str(ln)] for ln in bundle["tr_lineages"]]),
        np.vstack([get_pert_emb(p) for p in bundle["tr_perts"]]),
    ]).astype(np.float32)
    X_ev = np.hstack([
        np.repeat(cell_embs["RPE1"][None, :], len(bundle["ev_keys"]), axis=0),
        np.vstack([get_pert_emb(p) for p in bundle["ev_keys"]]),
    ]).astype(np.float32)

    # Fit 2-layer perturbation adapter + Ridge on frozen Geneformer-V2 representations
    torch.manual_seed(SEED)
    adapter = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(256, len(target_genes))
    ).to(DEVICE)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(bundle["Y_tr"])), batch_size=256, shuffle=True)
    adapter.train()
    for _ in range(30):
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(adapter(bx), by).backward()
            opt.step()
    adapter.eval()
    with torch.no_grad():
        pred = adapter(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)
    return pred


def run_scfoundation(bundle: dict) -> np.ndarray:
    print("\n--- [SOTA FM 2/2] scFoundation-100M ---")
    gene_idx_df = pd.read_csv(os.path.join(SCF_DIR, "OS_scRNA_gene_index.19264.tsv"), sep="\t")
    sym_col = gene_idx_df.columns[0]
    idx_col = gene_idx_df.columns[1] if len(gene_idx_df.columns) > 1 else None
    scf_genes = [str(g).strip().upper() for g in gene_idx_df[sym_col].tolist()]
    scf_g2i = {g: i for i, g in enumerate(scf_genes)}

    ckpt = torch.load(os.path.join(SCF_DIR, "model.pt"), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    # Locate the 19264+ gene embedding matrix (pos_emb / gene_emb)
    gene_emb_mat = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] >= 19264 and v.shape[1] in (512, 768):
            gene_emb_mat = v.detach().float().numpy()
            print(f"  Found scFoundation gene embedding '{k}': shape={gene_emb_mat.shape}")
            break
    if gene_emb_mat is None:
        raise RuntimeError("Could not locate 19264xD gene embedding matrix in scFoundation model.pt")

    d_emb = gene_emb_mat.shape[1]
    def get_scf_emb(p):
        idx = scf_g2i.get(str(p).upper(), -1)
        return gene_emb_mat[idx].astype(np.float32) if idx >= 0 else np.zeros(d_emb, dtype=np.float32)

    ctrl_map = {str(k): v for k, v in zip(bundle["ctrl_names"], bundle["ctrl_vecs"])}
    svd = TruncatedSVD(n_components=32, random_state=SEED)
    tr_ctrl_mat = np.vstack([ctrl_map[str(ln)] for ln in bundle["tr_lineages"]])
    z_tr = svd.fit_transform(tr_ctrl_mat).astype(np.float32)
    z_ev = svd.transform(np.repeat(ctrl_map["RPE1"][None, :], len(bundle["ev_keys"]), axis=0)).astype(np.float32)

    X_tr = np.hstack([z_tr, np.vstack([get_scf_emb(p) for p in bundle["tr_perts"]])]).astype(np.float32)
    X_ev = np.hstack([z_ev, np.vstack([get_scf_emb(p) for p in bundle["ev_keys"]])]).astype(np.float32)

    torch.manual_seed(SEED)
    adapter = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(256, len(bundle["target_genes"]))
    ).to(DEVICE)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(bundle["Y_tr"])), batch_size=256, shuffle=True)
    adapter.train()
    for _ in range(30):
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(adapter(bx), by).backward()
            opt.step()
    adapter.eval()
    with torch.no_grad():
        pred = adapter(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)
    return pred


def run_aivc_v11_and_ensemble(bundle: dict) -> tuple:
    print("\n--- [AIVC v1.1 & 5-Member Deep Ensemble] ---")
    target_genes = [str(g) for g in bundle["target_genes"]]
    n_genes = len(target_genes)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    tr_perts = [str(p) for p in bundle["tr_perts"]]
    tr_lineages = np.array([str(ln) for ln in bundle["tr_lineages"]])
    Y_tr = bundle["Y_tr"]

    bipartite = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)

    lineage_ctrls = {str(k): v for k, v in zip(bundle["ctrl_names"], bundle["ctrl_vecs"])}
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    pred_ridge_ev = np.load(os.path.join(OUT_DIR, "pred_Linear_Ridge.npy"))
    tr_vocab = sorted(set(tr_perts))
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    X_tr_lin = np.zeros((len(tr_perts), len(tr_vocab)), dtype=np.float32)
    for ri, p in enumerate(tr_perts):
        X_tr_lin[ri, lin_idx[p]] = 1.0
    pred_ridge_tr = np.zeros_like(Y_tr)
    for ln in np.unique(tr_lineages):
        m_out = (tr_lineages == ln)
        m_in = ~m_out
        if m_in.sum() > 0:
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[m_in], Y_tr[m_in])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out]).astype(np.float32)

    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U_prog = svd.fit(Y_tr).components_.astype(np.float32)
    causal_gene_mat = ((A_signed + 0.5 * (A_signed @ A_signed)) @ U_prog.T).astype(np.float32)

    def build_features(plist, lnames, ridge_preds, esm_mat, coexp_mat):
        ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
        for idx, (ln, p) in enumerate(zip(lnames, plist)):
            mc = lineage_ctrls[ln]
            ctrl_l.append(mc)
            teach_l.append(np.concatenate([coexp_mat[idx], esm_mat[idx]]))
            gi = gene_to_idx.get(p, -1)
            g_prot, g_phos = prot_phos_map[ln]
            if gi >= 0:
                raw_e = float(mc[gi])
                z_e = float((mc[gi] - mc.mean()) / (mc.std() + 1e-6))
                pr_v, ph_v = float(g_prot[gi]), float(g_phos[gi])
                c_vec = causal_gene_mat[gi]
            else:
                raw_e, z_e, pr_v, ph_v = 0.0, 0.0, 0.0, 0.0
                c_vec = np.zeros(64, dtype=np.float32)
            rn = float(np.log1p(np.linalg.norm(ridge_preds[idx])))
            tstat_l.append([raw_e, z_e, pr_v, ph_v, rn])
            causal_l.append(c_vec)
        return (
            np.asarray(ctrl_l, dtype=np.float32),
            np.asarray(teach_l, dtype=np.float32),
            ridge_preds.astype(np.float32),
            np.asarray(tstat_l, dtype=np.float32),
            np.asarray(causal_l, dtype=np.float32),
        )

    ctrl_tr, teach_tr, r_tr, tstat_tr, caus_tr = build_features(tr_perts, tr_lineages.tolist(), pred_ridge_tr, bundle["X_esm_tr"], bundle["X_coexp_tr"])
    ctrl_ev, teach_ev, r_ev, tstat_ev, caus_ev = build_features(ev_keys, ["RPE1"] * len(ev_keys), pred_ridge_ev, bundle["X_esm_ev"], bundle["X_coexp_ev"])

    seeds = [42, 101, 202, 303, 404]
    preds = []
    for sd_val in seeds:
        torch.manual_seed(sd_val)
        np.random.seed(sd_val)
        model = LineageInvariantPathwayNet(n_genes, teach_tr.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
        dl = DataLoader(
            TensorDataset(
                torch.tensor(ctrl_tr), torch.tensor(teach_tr), torch.tensor(r_tr),
                torch.tensor(tstat_tr), torch.tensor(caus_tr), torch.tensor(Y_tr),
            ),
            batch_size=512, shuffle=True
        )
        model.train()
        for _ in range(30):
            for bc, bt, br, bts, bca, by in dl:
                bc, bt, br, bts, bca, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bts.to(DEVICE), bca.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                out = model(bc, bt, br, bts, bca)
                F.mse_loss(out, by).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            pev = model(
                torch.tensor(ctrl_ev, device=DEVICE), torch.tensor(teach_ev, device=DEVICE),
                torch.tensor(r_ev, device=DEVICE), torch.tensor(tstat_ev, device=DEVICE),
                torch.tensor(caus_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        preds.append(pev)
    return preds[0], np.mean(preds, axis=0).astype(np.float32)


def main():
    t0 = time.time()
    bundle = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    ev_keys = [str(k) for k in bundle["ev_keys"]]
    d_rpe1 = {ev_keys[i]: bundle["Y_ev"][i] for i in range(len(ev_keys))}

    p_gf = run_geneformer_v2(bundle, d_rpe1)
    np.save(os.path.join(OUT_DIR, "pred_Geneformer_V2_104M.npy"), p_gf)

    p_scf = run_scfoundation(bundle)
    np.save(os.path.join(OUT_DIR, "pred_scFoundation_100M.npy"), p_scf)

    p_aivc, p_ens = run_aivc_v11_and_ensemble(bundle)
    np.save(os.path.join(OUT_DIR, "pred_AIVC_v1.1.npy"), p_aivc)
    np.save(os.path.join(OUT_DIR, "pred_AIVC_v1.1_Ensemble.npy"), p_ens)

    for mname, pmat in [
        ("Geneformer_V2_104M", p_gf),
        ("scFoundation_100M", p_scf),
        ("AIVC_v1.1", p_aivc),
        ("AIVC_v1.1_Ensemble", p_ens),
    ]:
        df_m = evaluate_perturbation_predictions(d_rpe1, {ev_keys[i]: pmat[i] for i in range(len(ev_keys))}, ev_keys)
        print(f"  [{mname:20s}] MeanR={df_m['pearson_delta'].mean():.4f} | Top20R={df_m['pearson_delta_top20'].mean():.4f} | DirAcc={df_m['dir_acc_top20'].mean():.4f} | MSE={df_m['mse'].mean():.6f}")
    print(f"Part 2 completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
