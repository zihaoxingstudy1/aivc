# -*- coding: utf-8 -*-
"""
exp46_fair_equal_data_norman_overexpr.py
Strict Equal-Data Fairness Test on Norman2019 Single-Gene Overexpression (21 unseen genes):
  Regime A (Norman-Only, n=84): All models trained strictly on Norman 84 genes.
  Regime B (+Joung 485 ORFs Equal-Data, n=569): ALL models (Linear-Ridge, Static-MLP,
    Geneformer-V2-104M, scFoundation-100M, and AIVC_Overexpr_CisOmniPath) given the
    EXACT same 485 Joung ORFs + 84 Norman genes (strictly excluding all 21 test genes).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import evaluate_perturbation_predictions
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp3_triteacher_distill import load_esm2_gene_embeddings
from exp37c_overexpression_cis_omnipath import CisOmniPathOverexprModel
from sota_bench_norman_part1_fm_and_aivc import extract_gf_and_scf_gene_embs
from eval_calibrated_wmse_pds import compute_wmse_per_pert, compute_pds_per_pert

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\data\AIVC\data"
PRIORS_DIR = r"D:\data\AIVC\data\priors"
BUNDLE_NPZ = r"D:\data\AIVC\test_results\sota_norman_preds\norman_ood_bundle.npz"
JOUNG_NPZ = r"D:\data\AIVC\data\joung_orf_norman_hvg2000_cache.npz"
OUT_CSV = r"D:\data\AIVC\test_results\exp46_fair_equal_data_norman_overexpr.csv"


from scipy.stats import pearsonr

def eval_all(name, regime, y_true, y_pred):
    p_all, p_top20, d_top20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p_all.append(float(pearsonr(yt, yp)[0]))
        idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[idx]) > 1e-6 and np.std(yp[idx]) > 1e-6:
            p_top20.append(float(pearsonr(yt[idx], yp[idx])[0]))
        d_top20.append(float((np.sign(yt[idx]) == np.sign(yp[idx])).mean()))
    wmse = float(np.mean(compute_wmse_per_pert(y_true, y_pred)))
    pds = float(np.mean(compute_pds_per_pert(y_true, y_pred)))
    return {
        "Regime": regime, "Model": name,
        "MeanR": float(np.mean(p_all)),
        "Top20R": float(np.mean(p_top20)),
        "DirAcc": float(np.mean(d_top20)),
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "WMSE": wmse, "PDS": pds,
    }


def fit_mlp_probe(X_tr, Y_tr, X_ev, n_genes, seed=42, epochs=90):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(192, n_genes)
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=3e-3)
    xt, yt = torch.tensor(X_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(xt)
        loss = F.mse_loss(pred, yt) + 0.05 * (1.0 - F.cosine_similarity(pred, yt, dim=-1)).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)


def train_aivc_overexpr(genes_tr, Y_tr, genes_ev, target_genes, g2i, A_signed, esm_map, n_genes, pretrain_genes=None, pretrain_Y=None):
    k_prog = 32
    svd = TruncatedSVD(n_components=k_prog, random_state=42)
    U_prog = svd.fit(Y_tr).components_.astype(np.float32)
    A_s2 = A_signed @ A_signed
    causal_mat = ((A_signed + 0.5 * A_s2) @ U_prog.T).astype(np.float32)

    def get_feats(glist):
        e_mat = np.vstack([esm_map.get(g.upper(), np.zeros(640, np.float32)) for g in glist]).astype(np.float32)
        c_mat = np.zeros((len(glist), k_prog), dtype=np.float32)
        cis_mat = np.zeros((len(glist), n_genes), dtype=np.float32)
        op1_mat = np.zeros((len(glist), n_genes), dtype=np.float32)
        op2_mat = np.zeros((len(glist), n_genes), dtype=np.float32)
        for idx, g in enumerate(glist):
            gi = g2i.get(g.upper(), -1)
            if gi >= 0:
                c_mat[idx] = causal_mat[gi]
                cis_mat[idx, gi] = 1.0
                op1_mat[idx] = A_signed[:, gi]
                op2_mat[idx] = A_s2[:, gi]
        return e_mat, c_mat, cis_mat, op1_mat, op2_mat

    E_tr, C_tr, Cis_tr, Op1_tr, Op2_tr = get_feats(genes_tr)
    E_ev, C_ev, Cis_ev, Op1_ev, Op2_ev = get_feats(genes_ev)
    r_oof = np.zeros_like(Y_tr)
    for i in range(len(genes_tr)):
        m = np.arange(len(genes_tr)) != i
        r_oof[i] = Ridge(alpha=20.0).fit(E_tr[m], Y_tr[m]).predict(E_tr[i:i+1])[0]
    r_ev = Ridge(alpha=20.0).fit(E_tr, Y_tr).predict(E_ev).astype(np.float32)

    preds = []
    for sd in [42, 101, 202, 303, 404]:
        torch.manual_seed(sd)
        np.random.seed(sd)
        net = CisOmniPathOverexprModel(640, k_prog, U_prog).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=3e-3)
        if pretrain_genes is not None and pretrain_Y is not None:
            E_p, C_p, Cis_p, Op1_p, Op2_p = get_feats(pretrain_genes)
            xp = torch.tensor(E_p, device=DEVICE)
            cp = torch.tensor(C_p, device=DEVICE)
            cisp = torch.tensor(Cis_p, device=DEVICE)
            op1p = torch.tensor(Op1_p, device=DEVICE)
            op2p = torch.tensor(Op2_p, device=DEVICE)
            yp = torch.tensor(pretrain_Y, device=DEVICE)
            net.train()
            for _ in range(40):
                opt.zero_grad()
                pr = net.forward_mech(xp, cp, cisp, op1p, op2p)
                loss = F.mse_loss(pr, yp) + 0.05 * (1.0 - F.cosine_similarity(pr, yp, dim=-1)).mean()
                loss.backward()
                opt.step()
        xt = torch.tensor(E_tr, device=DEVICE)
        ct = torch.tensor(C_tr, device=DEVICE)
        rt = torch.tensor(r_oof, device=DEVICE)
        cist = torch.tensor(Cis_tr, device=DEVICE)
        op1t = torch.tensor(Op1_tr, device=DEVICE)
        op2t = torch.tensor(Op2_tr, device=DEVICE)
        yt = torch.tensor(Y_tr, device=DEVICE)
        net.train()
        for _ in range(60):
            opt.zero_grad()
            pr = net(xt, ct, rt, cist, op1t, op2t)
            loss = F.mse_loss(pr, yt) + 0.05 * (1.0 - F.cosine_similarity(pr, yt, dim=-1)).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pev = net(
                torch.tensor(E_ev, device=DEVICE), torch.tensor(C_ev, device=DEVICE),
                torch.tensor(r_ev, device=DEVICE), torch.tensor(Cis_ev, device=DEVICE),
                torch.tensor(Op1_ev, device=DEVICE), torch.tensor(Op2_ev, device=DEVICE),
            ).cpu().numpy().astype(np.float32)
        preds.append(pev)
    return np.mean(preds, axis=0).astype(np.float32)


def main():
    print("=== Exp46: Strict Equal-Data Fairness Test (Norman2019 Single Overexpression, 84->21) ===")
    b = np.load(BUNDLE_NPZ, allow_pickle=True)
    target_genes = [str(g).upper() for g in b["target_genes"]]
    n_genes = len(target_genes)
    g2i = {g: i for i, g in enumerate(target_genes)}
    tr_s = [str(g).upper() for g in b["tr_s"]]
    ev_s = [str(g).upper() for g in b["ev_s"]]
    Ys_tr = b["Ys_tr"].astype(np.float32)
    Ys_ev = b["Ys_ev"].astype(np.float32)

    jk = np.load(JOUNG_NPZ, allow_pickle=True)
    ev_set = set(ev_s)
    keep_j = [i for i, g in enumerate(jk["orf_genes"]) if str(g).upper() not in ev_set]
    j_genes = [str(jk["orf_genes"][i]).upper() for i in keep_j]
    j_Y = jk["orf_deltas"][keep_j].astype(np.float32)
    print(f"Loaded Norman train={len(tr_s)}, test={len(ev_s)}, Joung zero-leakage ORFs={len(j_genes)}")

    all_genes = sorted(set(tr_s) | set(ev_s) | set(j_genes))
    esm_map = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    esm_map = {k.upper(): v for k, v in esm_map.items()}
    gf_map, scf_map = extract_gf_and_scf_gene_embs(all_genes)
    bip = build_gene_pathway_bipartite_tensors(target_genes, PRIORS_DIR, min_genes_per_pathway=3, include_biogrid=False)
    A_signed = (bip["A_pos"] - bip["A_neg"]).astype(np.float32)

    def mat_from(glist, mp, dim):
        return np.vstack([mp.get(g, np.zeros(dim, np.float32)) for g in glist]).astype(np.float32)

    E_tr, E_ev, E_j = mat_from(tr_s, esm_map, 640), mat_from(ev_s, esm_map, 640), mat_from(j_genes, esm_map, 640)
    G_tr, G_ev, G_j = mat_from(tr_s, gf_map, 768), mat_from(ev_s, gf_map, 768), mat_from(j_genes, gf_map, 768)
    S_tr, S_ev, S_j = mat_from(tr_s, scf_map, 768), mat_from(ev_s, scf_map, 768), mat_from(j_genes, scf_map, 768)

    rows = []
    # --- Regime A: Norman-Only (n=84) ---
    regA = "A_Norman_Only(84)"
    rows.append(eval_all("Linear-Ridge (ESM2)", regA, Ys_ev, Ridge(alpha=20.0).fit(E_tr, Ys_tr).predict(E_ev)))
    rows.append(eval_all("Static-MLP (ESM2)", regA, Ys_ev, fit_mlp_probe(E_tr, Ys_tr, E_ev, n_genes)))
    rows.append(eval_all("Geneformer-V2-104M", regA, Ys_ev, fit_mlp_probe(G_tr, Ys_tr, G_ev, n_genes)))
    rows.append(eval_all("scFoundation-100M", regA, Ys_ev, fit_mlp_probe(S_tr, Ys_tr, S_ev, n_genes)))
    rows.append(eval_all("AIVC_Overexpr_CisOmniPath_Ens5", regA, Ys_ev, train_aivc_overexpr(tr_s, Ys_tr, ev_s, target_genes, g2i, A_signed, esm_map, n_genes)))

    # --- Regime B: Equal-Data +Joung 485 ORFs (n=569) ---
    regB = "B_EqualData_Plus_Joung(569)"
    E_tr_all, Y_all = np.vstack([E_j, E_tr]), np.vstack([j_Y, Ys_tr])
    G_tr_all = np.vstack([G_j, G_tr])
    S_tr_all = np.vstack([S_j, S_tr])

    rows.append(eval_all("Linear-Ridge (ESM2 + Joung)", regB, Ys_ev, Ridge(alpha=30.0).fit(E_tr_all, Y_all).predict(E_ev)))
    rows.append(eval_all("Static-MLP (ESM2 + Joung)", regB, Ys_ev, fit_mlp_probe(E_tr_all, Y_all, E_ev, n_genes)))
    rows.append(eval_all("Geneformer-V2-104M (+Joung)", regB, Ys_ev, fit_mlp_probe(G_tr_all, Y_all, G_ev, n_genes)))
    rows.append(eval_all("scFoundation-100M (+Joung)", regB, Ys_ev, fit_mlp_probe(S_tr_all, Y_all, S_ev, n_genes)))
    rows.append(eval_all("AIVC_Overexpr_CisOmniPath_Ens5 (+Joung)", regB, Ys_ev,
                         train_aivc_overexpr(tr_s, Ys_tr, ev_s, target_genes, g2i, A_signed, esm_map, n_genes, pretrain_genes=j_genes, pretrain_Y=j_Y)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
