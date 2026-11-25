# -*- coding: utf-8 -*-
"""
sota_bench_norman_part1_fm_and_aivc.py
Export exact Norman2019 Single-Gene Overexpression (84->21) & Double-Gene Combo (104->27) OOD bundles,
and run Geneformer-V2-104M, scFoundation-100M, Linear-Ridge, Additive, and AIVC (Exp37c / Exp39).
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp3_triteacher_distill import load_esm2_gene_embeddings
from exp37c_overexpression_cis_omnipath import CisOmniPathOverexprModel
from exp39_norman_combo_epistasis_benchmark import ComboEpistasisOmniPathNet, eval_epistasis

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
GF_DIR = r"D:\data\AIVC\pretrained\geneformer"
SCF_DIR = r"D:\data\AIVC\pretrained\scfoundation"
OUT_DIR = r"D:\data\AIVC\test_results\sota_norman_preds"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_gf_and_scf_gene_embs(all_pert_genes: list) -> tuple:
    # 1. Geneformer-V2-104M token embeddings (768d)
    with open(os.path.join(GF_DIR, "geneformer", "gene_name_id_dict_gc104M.pkl"), "rb") as f:
        g2ens = pickle.load(f)
    with open(os.path.join(GF_DIR, "geneformer", "token_dictionary_gc104M.pkl"), "rb") as f:
        ens2tok = pickle.load(f)
    gf = BertModel.from_pretrained(os.path.join(GF_DIR, "Geneformer-V2-104M_CLcancer"))
    w_gf = gf.embeddings.word_embeddings.weight.detach().float().numpy()
    del gf
    gf_map = {}
    for g in all_pert_genes:
        ens = g2ens.get(g) or g2ens.get(g.upper())
        tid = ens2tok.get(ens, -1) if ens is not None else -1
        gf_map[g.upper()] = w_gf[tid].astype(np.float32) if tid >= 0 else np.zeros(w_gf.shape[1], dtype=np.float32)

    # 2. scFoundation-100M gene embeddings (768d)
    gdf = pd.read_csv(os.path.join(SCF_DIR, "OS_scRNA_gene_index.19264.tsv"), sep="\t")
    scf_g2i = {str(g).strip().upper(): i for i, g in enumerate(gdf[gdf.columns[0]].tolist())}
    ck = torch.load(os.path.join(SCF_DIR, "model.pt"), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    w_scf = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] >= 19264 and v.shape[1] in (512, 768):
            w_scf = v.detach().float().numpy()
            break
    scf_map = {}
    for g in all_pert_genes:
        idx = scf_g2i.get(g.upper(), -1)
        scf_map[g.upper()] = w_scf[idx].astype(np.float32) if idx >= 0 else np.zeros(w_scf.shape[1], dtype=np.float32)
    return gf_map, scf_map


def fit_adapter(X_tr, Y_tr, X_ev, n_genes, epochs=80):
    torch.manual_seed(SEED)
    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(192, n_genes)
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=3e-3)
    xt, yt = torch.tensor(X_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(net(xt), yt).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)


def main():
    t0 = time.time()
    print(f"=== SOTA Benchmark on Norman2019 (Overexpr + Combo) Part 1 ({DEVICE}) ===")
    norm_ad = sc.read_h5ad(os.path.join(DATA_DIR, "Norman2019_hvg2000.h5ad"))
    target_genes = [str(g) for g in norm_ad.var_names]
    n_genes = len(target_genes)
    mu_ctrl = _to_dense_array(norm_ad[norm_ad.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_norm = _compute_group_deltas(norm_ad[norm_ad.obs["perturbation"] != "control"], "perturbation", mu_ctrl)
    del norm_ad

    singles = {p: v for p, v in d_norm.items() if "_" not in p and "+" not in p}
    single_perts = sorted(singles.keys())
    rng = np.random.RandomState(SEED)
    perm_s = rng.permutation(single_perts)
    tr_s = sorted(perm_s[:int(0.8 * len(single_perts))].tolist())
    ev_s = sorted(perm_s[int(0.8 * len(single_perts)):].tolist())
    Ys_tr = np.vstack([singles[p] for p in tr_s]).astype(np.float32)
    Ys_ev = np.vstack([singles[p] for p in ev_s]).astype(np.float32)

    combos_valid = sorted(
        [(p, p.split("_")[0], p.split("_")[1], v) for p, v in d_norm.items() if "_" in p and len(p.split("_")) == 2 and p.split("_")[0] in singles and p.split("_")[1] in singles],
        key=lambda x: x[0]
    )
    rng2 = np.random.RandomState(SEED)
    perm_c = rng2.permutation(len(combos_valid))
    tr_c = [combos_valid[i] for i in perm_c[:int(0.8 * len(combos_valid))]]
    ev_c = [combos_valid[i] for i in perm_c[int(0.8 * len(combos_valid)):]]
    Yc_tr = np.vstack([c[3] for c in tr_c]).astype(np.float32)
    Yc_ev = np.vstack([c[3] for c in ev_c]).astype(np.float32)
    Xadd_c_ev = np.vstack([singles[c[1]] + singles[c[2]] for c in ev_c]).astype(np.float32)

    # Save bundle for bench_sota (scGPT + GEARS-SGConv)
    np.savez_compressed(
        os.path.join(OUT_DIR, "norman_ood_bundle.npz"),
        target_genes=np.array(target_genes),
        mu_ctrl=mu_ctrl,
        tr_s=np.array(tr_s), ev_s=np.array(ev_s), Ys_tr=Ys_tr, Ys_ev=Ys_ev,
        tr_c_names=np.array([c[0] for c in tr_c]), ev_c_names=np.array([c[0] for c in ev_c]),
        tr_c_gA=np.array([c[1] for c in tr_c]), tr_c_gB=np.array([c[2] for c in tr_c]),
        ev_c_gA=np.array([c[1] for c in ev_c]), ev_c_gB=np.array([c[2] for c in ev_c]),
        Yc_tr=Yc_tr, Yc_ev=Yc_ev, Xadd_c_ev=Xadd_c_ev,
    )

    all_g = sorted(set(single_perts) | {c[1] for c in combos_valid} | {c[2] for c in combos_valid})
    gf_map, scf_map = extract_gf_and_scf_gene_embs(all_g)

    for tag, emb_map in [("Geneformer_V2_104M", gf_map), ("scFoundation_100M", scf_map)]:
        # Single-gene overexpression
        xs_tr = np.vstack([emb_map[p.upper()] for p in tr_s])
        xs_ev = np.vstack([emb_map[p.upper()] for p in ev_s])
        ps = fit_adapter(xs_tr, Ys_tr, xs_ev, n_genes, epochs=80)
        np.save(os.path.join(OUT_DIR, f"single_pred_{tag}.npy"), ps)

        # Double-gene combo (symmetric sum + elementwise product)
        xc_tr = np.vstack([np.concatenate([emb_map[c[1].upper()] + emb_map[c[2].upper()], emb_map[c[1].upper()] * emb_map[c[2].upper()]]) for c in tr_c])
        xc_ev = np.vstack([np.concatenate([emb_map[c[1].upper()] + emb_map[c[2].upper()], emb_map[c[1].upper()] * emb_map[c[2].upper()]]) for c in ev_c])
        pc = fit_adapter(xc_tr, Yc_tr, xc_ev, n_genes, epochs=90)
        np.save(os.path.join(OUT_DIR, f"combo_pred_{tag}.npy"), pc)

        ds_df = evaluate_perturbation_predictions({ev_s[i]: Ys_ev[i] for i in range(len(ev_s))}, {ev_s[i]: ps[i] for i in range(len(ev_s))}, tag, [20])
        dc_df = evaluate_perturbation_predictions({ev_c[i][0]: Yc_ev[i] for i in range(len(ev_c))}, {ev_c[i][0]: pc[i] for i in range(len(ev_c))}, tag, [20])
        epi_r, _ = eval_epistasis(Yc_ev, pc, Xadd_c_ev)
        print(f"  [{tag:20s}] Single Top20R={ds_df['pearson_delta_top20'].mean():.4f} | Combo Top20R={dc_df['pearson_delta_top20'].mean():.4f} | EpistasisR={epi_r:.4f}")

    print(f"Part 1 completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
