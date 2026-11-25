# -*- coding: utf-8 -*-
"""
exp50_combo_component_ablation.py
What drives AIVC_Combo_Epistasis on Norman2019 unseen double-gene combinations (104 train -> 27 test)?
pred = w_add*(dA+dB) + w_dom*dominant + alpha_xtalk*(opA*opB) + 0.35*(head(feats) @ U_prog)
Arms (5 seeds each; same split/optimizer as exp39):
  full_signed, no_xtalk, xtalk_unsigned (|A| instead of signed A), no_dom, no_feat_input (head sees zeros),
  no_esm (ESM-2 part of feats zeroed), no_zfeat (single-gene program part zeroed), additive_gain_only
Paired bootstrap over the 27 test combinations (5,000 resamples) of the MSE difference vs full_signed.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from lbb import _compute_group_deltas, _to_dense_array
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, r"D:\data\AIVC\test_python_scripts")
from exp3_triteacher_distill import load_esm2_gene_embeddings
from exp39_norman_combo_epistasis_benchmark import ComboEpistasisOmniPathNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = r"D:\data\AIVC\data"
PRIORS = r"D:\data\AIVC\data\priors"
OUT_CSV = r"D:\data\AIVC\test_results\exp50_combo_component_ablation.csv"
OUT_PAIRED = r"D:\data\AIVC\test_results\exp50_combo_paired_bootstrap.csv"
SEED = 42


def mets(yt, yp, yadd):
    pa, p20, dd, epi = [], [], [], []
    for i in range(len(yt)):
        if np.std(yt[i]) > 1e-6 and np.std(yp[i]) > 1e-6:
            pa.append(pearsonr(yt[i], yp[i])[0])
        idx = np.argsort(-np.abs(yt[i]))[:20]
        if np.std(yt[i][idx]) > 1e-6 and np.std(yp[i][idx]) > 1e-6:
            p20.append(pearsonr(yt[i][idx], yp[i][idx])[0])
        dd.append((np.sign(yt[i][idx]) == np.sign(yp[i][idx])).mean())
        te, pe = yt[i] - yadd[i], yp[i] - yadd[i]
        if np.std(te) > 1e-6 and np.std(pe) > 1e-6:
            epi.append(pearsonr(te, pe)[0])
    return {"MeanR": float(np.mean(pa)), "Top20R": float(np.mean(p20)), "DirAcc": float(np.mean(dd)),
            "MSE": float(np.mean((yt - yp) ** 2)), "EpistasisR": float(np.mean(epi))}


def main():
    ad = sc.read_h5ad(os.path.join(DATA, "Norman2019_hvg2000.h5ad"))
    genes = [str(g) for g in ad.var_names]
    G = len(genes)
    g2i = {g.upper(): i for i, g in enumerate(genes)}
    mu = _to_dense_array(ad[ad.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d = _compute_group_deltas(ad[ad.obs["perturbation"] != "control"], "perturbation", mu)
    del ad
    singles = {p: v for p, v in d.items() if "_" not in p and "+" not in p}
    combos = []
    for p, v in d.items():
        if "_" in p:
            parts = p.split("_")
            if len(parts) == 2 and parts[0] in singles and parts[1] in singles:
                combos.append((p, parts[0], parts[1], v))
    combos.sort(key=lambda x: x[0])
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(combos))
    ntr = int(0.8 * len(combos))
    tr = [combos[i] for i in perm[:ntr]]
    ev = [combos[i] for i in perm[ntr:]]
    print(f"singles={len(singles)} combos={len(combos)} train={len(tr)} test={len(ev)}")
    esm = {str(k).upper(): v for k, v in load_esm2_gene_embeddings(os.path.join(DATA, "target_esm2_150m.npz")).items()}
    d_esm = len(next(iter(esm.values())))
    bip = build_gene_pathway_bipartite_tensors(genes, PRIORS, min_genes_per_pathway=3, include_biogrid=False)
    A_s = (bip["A_pos"] - bip["A_neg"]).astype(np.float32)
    A_u = np.abs(bip["A_pos"] - bip["A_neg"]).astype(np.float32)  # unsigned: edge presence only
    Y_tr = np.vstack([c[3] for c in tr]).astype(np.float32)
    Y_ev = np.vstack([c[3] for c in ev]).astype(np.float32)
    U = TruncatedSVD(n_components=32, random_state=SEED).fit(np.vstack(list(singles.values()) + [Y_tr])).components_.astype(np.float32)

    def build(clist, A):
        xa, xd, xf, xx = [], [], [], []
        for _, a, b_, _ in clist:
            dA, dB = singles[a], singles[b_]
            xa.append(dA + dB)
            xd.append(np.where(np.abs(dA) >= np.abs(dB), dA, dB))
            eA = esm.get(a.upper(), np.zeros(d_esm, np.float32)); eB = esm.get(b_.upper(), np.zeros(d_esm, np.float32))
            zA, zB = dA @ U.T, dB @ U.T
            xf.append(np.concatenate([zA + zB, zA * zB, eA + eB, eA * eB]))
            gA, gB = g2i.get(a.upper(), -1), g2i.get(b_.upper(), -1)
            oA = A[:, gA] if gA >= 0 else np.zeros(G, np.float32)
            oB = A[:, gB] if gB >= 0 else np.zeros(G, np.float32)
            xx.append(oA * oB)
        return [np.asarray(x, np.float32) for x in (xa, xd, xf, xx)]

    base_tr, base_ev = build(tr, A_s), build(ev, A_s)
    uns_tr, uns_ev = build(tr, A_u), build(ev, A_u)
    print("Test combos with non-zero OmniPath cross-talk vector:", int((np.abs(base_ev[3]).sum(1) > 0).sum()), "/", len(ev),
          "| train:", int((np.abs(base_tr[3]).sum(1) > 0).sum()), "/", len(tr))
    zdim = 2 * 32

    def arm_inputs(arm):
        a_tr, a_ev = [x.copy() for x in base_tr], [x.copy() for x in base_ev]
        if arm == "xtalk_unsigned":
            a_tr, a_ev = [x.copy() for x in uns_tr], [x.copy() for x in uns_ev]
        for sets in (a_tr, a_ev):
            if arm in ("no_xtalk", "additive_gain_only"):
                sets[3][:] = 0
            if arm in ("no_dom", "additive_gain_only"):
                sets[1][:] = 0
            if arm in ("no_feat_input", "additive_gain_only"):
                sets[2][:] = 0
            if arm == "no_esm":
                sets[2][:, zdim:] = 0
            if arm == "no_zfeat":
                sets[2][:, :zdim] = 0
        return a_tr, a_ev

    arms = ["full_signed", "no_xtalk", "xtalk_unsigned", "no_dom", "no_feat_input", "no_esm", "no_zfeat", "additive_gain_only"]
    rows, ens_preds = [], {}
    yadd = base_ev[0]
    for arm in arms:
        a_tr, a_ev = arm_inputs(arm)
        preds, seed_mse = [], []
        for sd in [42, 101, 202, 303, 404]:
            torch.manual_seed(sd)
            net = ComboEpistasisOmniPathNet(a_tr[2].shape[1], 32, U, hid=128).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=5e-3)
            T = [torch.tensor(x, device=DEVICE) for x in a_tr]
            yt = torch.tensor(Y_tr, device=DEVICE)
            net.train()
            for _ in range(90):
                opt.zero_grad()
                pr = net(*T)
                loss = F.mse_loss(pr, yt) + 0.03 * (1.0 - F.cosine_similarity(pr, yt, dim=-1)).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pe = net(*[torch.tensor(x, device=DEVICE) for x in a_ev]).cpu().numpy().astype(np.float32)
            preds.append(pe)
            seed_mse.append(float(np.mean((Y_ev - pe) ** 2)))
        pens = np.mean(preds, axis=0).astype(np.float32)
        ens_preds[arm] = pens
        m = mets(Y_ev, pens, yadd)
        m.update({"Arm": arm, "Seed5_MSE_Mean": float(np.mean(seed_mse)), "Seed5_MSE_SD": float(np.std(seed_mse, ddof=1))})
        rows.append(m)
        print(f"[{arm:18s}] MeanR={m['MeanR']:.4f} Top20={m['Top20R']:.4f} MSE={m['MSE']:.5f} EpiR={m['EpistasisR']:.4f} | 5seed MSE {m['Seed5_MSE_Mean']:.5f}+/-{m['Seed5_MSE_SD']:.5f}", flush=True)
    df = pd.DataFrame(rows)[["Arm", "MeanR", "Top20R", "DirAcc", "MSE", "EpistasisR", "Seed5_MSE_Mean", "Seed5_MSE_SD"]]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    r2 = np.random.RandomState(1)
    idx = r2.randint(0, len(Y_ev), size=(5000, len(Y_ev)))
    per = {a: ((Y_ev - p) ** 2).mean(1) for a, p in ens_preds.items()}
    prow = []
    for a in arms[1:]:
        dlt = per[a] - per["full_signed"]
        bs = dlt[idx].mean(1)
        prow.append({"Arm_vs_full": a, "Delta_MSE(arm-full)": float(dlt.mean()), "CI95_lo": float(np.percentile(bs, 2.5)), "CI95_hi": float(np.percentile(bs, 97.5))})
    dp = pd.DataFrame(prow)
    dp.to_csv(OUT_PAIRED, index=False, encoding="utf-8-sig")
    print("\n" + df.round(5).to_string(index=False))
    print("\nPaired bootstrap over 27 test combos (positive Delta_MSE = arm worse than full):")
    print(dp.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
