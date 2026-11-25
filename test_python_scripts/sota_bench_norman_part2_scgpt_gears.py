# -*- coding: utf-8 -*-
"""
sota_bench_norman_part2_scgpt_gears.py
Run in bench_sota env: evaluates scGPT-WholeHuman and GEARS-SGConv on the exact same
Norman2019 Single-Gene Overexpression (84->21) & Double-Gene Combo (104->27) OOD splits,
and merges all open-source SOTA models + AIVC (Exp37c/Exp39) into unified CSV tables.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch_geometric.nn import SGConv

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = r"D:\data\AIVC\test_results\sota_norman_preds"
SCGPT_DIR = r"D:\data\AIVC\pretrained\scgpt"
OUT_SINGLE_CSV = r"D:\data\AIVC\test_results\sota_benchmark_norman_overexpr.csv"
OUT_COMBO_CSV = r"D:\data\AIVC\test_results\sota_benchmark_norman_combo.csv"


def eval_metrics(y_true, y_pred, y_add=None):
    p_all, p_top20, d_top20 = [], [], []
    for i in range(len(y_true)):
        yt, yp = y_true[i], y_pred[i]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            p_all.append(float(pearsonr(yt, yp)[0]))
        top_idx = np.argsort(-np.abs(yt))[:20]
        if np.std(yt[top_idx]) > 1e-6 and np.std(yp[top_idx]) > 1e-6:
            p_top20.append(float(pearsonr(yt[top_idx], yp[top_idx])[0]))
        d_top20.append(float((np.sign(yt[top_idx]) == np.sign(yp[top_idx])).mean()))
    res = {
        "mean_pearson": float(np.mean(p_all)),
        "top20_pearson": float(np.mean(p_top20)),
        "dir_acc_top20": float(np.mean(d_top20)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }
    if y_add is not None:
        te, pe = y_true - y_add, y_pred - y_add
        ers = [float(pearsonr(te[i], pe[i])[0]) for i in range(len(te)) if np.std(te[i]) > 1e-6 and np.std(pe[i]) > 1e-6]
        res["epistasis_r"] = float(np.mean(ers)) if ers else 0.0
    return res


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


class GEARSSGConvNorman(nn.Module):
    def __init__(self, n_genes: int, d_in: int, edge_index: torch.Tensor, hid: int = 64):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.gene_base = nn.Parameter(torch.randn(n_genes, hid) * 0.05)
        self.pert_proj = nn.Linear(d_in, hid)
        self.sgc = SGConv(hid, hid, K=2)
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        h_base = self.sgc(self.gene_base, self.edge_index)
        return self.readout(h_base.unsqueeze(0) + self.pert_proj(x).unsqueeze(1)).squeeze(-1)


def run_gears_sgconv(X_tr, Y_tr, X_ev, n_genes):
    Y_n = Y_tr - Y_tr.mean(axis=0, keepdims=True)
    Y_n = Y_n / (Y_n.std(axis=0, keepdims=True) + 1e-6)
    corr = (Y_n.T @ Y_n) / float(len(Y_tr))
    np.fill_diagonal(corr, 0.0)
    src_l, dst_l = [], []
    for i in range(n_genes):
        for j in np.argsort(-np.abs(corr[i]))[:15]:
            src_l.append(i); dst_l.append(int(j))
    edge_index = torch.tensor([src_l, dst_l], dtype=torch.long, device=DEVICE)

    torch.manual_seed(SEED)
    net = GEARSSGConvNorman(n_genes, X_tr.shape[1], edge_index, hid=64).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-3)
    xt, yt = torch.tensor(X_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
    net.train()
    for _ in range(80):
        opt.zero_grad()
        F.mse_loss(net(xt), yt).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)


def main():
    t0 = time.time()
    print(f"=== SOTA Benchmark on Norman2019 Part 2: scGPT + GEARS-SGConv ({DEVICE}) ===")
    b = np.load(os.path.join(OUT_DIR, "norman_ood_bundle.npz"), allow_pickle=True)
    n_genes = len(b["target_genes"])
    tr_s, ev_s = [str(x) for x in b["tr_s"]], [str(x) for x in b["ev_s"]]
    Ys_tr, Ys_ev = b["Ys_tr"], b["Ys_ev"]
    tr_c_gA, tr_c_gB = [str(x) for x in b["tr_c_gA"]], [str(x) for x in b["tr_c_gB"]]
    ev_c_gA, ev_c_gB = [str(x) for x in b["ev_c_gA"]], [str(x) for x in b["ev_c_gB"]]
    Yc_tr, Yc_ev, Xadd_c_ev = b["Yc_tr"], b["Yc_ev"], b["Xadd_c_ev"]

    # 1. scGPT pretrained gene embeddings (512d) via direct json + safetensors (zero scvi/lightning deadlock)
    with open(os.path.join(SCGPT_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sd = load_safetensors(os.path.join(SCGPT_DIR, "model.safetensors"))
    w_scgpt = None
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[0] == len(vocab) and v.shape[1] == 512:
            w_scgpt = v.detach().float().numpy()
            print(f"  Found scGPT embedding '{k}': {w_scgpt.shape}")
            break
    if w_scgpt is None:
        raise RuntimeError(f"Could not find ({len(vocab)}, 512) embedding in scGPT safetensors; keys={list(sd.keys())[:10]}")

    def g_emb(g):
        if g in vocab:
            return w_scgpt[vocab[g]].astype(np.float32)
        if g.upper() in vocab:
            return w_scgpt[vocab[g.upper()]].astype(np.float32)
        return np.zeros(w_scgpt.shape[1], dtype=np.float32)

    xs_tr = np.vstack([g_emb(g) for g in tr_s])
    xs_ev = np.vstack([g_emb(g) for g in ev_s])
    p_scgpt_s = fit_adapter(xs_tr, Ys_tr, xs_ev, n_genes, epochs=80)
    np.save(os.path.join(OUT_DIR, "single_pred_scGPT.npy"), p_scgpt_s)

    xc_tr = np.vstack([np.concatenate([g_emb(a) + g_emb(c), g_emb(a) * g_emb(c)]) for a, c in zip(tr_c_gA, tr_c_gB)])
    xc_ev = np.vstack([np.concatenate([g_emb(a) + g_emb(c), g_emb(a) * g_emb(c)]) for a, c in zip(ev_c_gA, ev_c_gB)])
    p_scgpt_c = fit_adapter(xc_tr, Yc_tr, xc_ev, n_genes, epochs=90)
    np.save(os.path.join(OUT_DIR, "combo_pred_scGPT.npy"), p_scgpt_c)

    # 2. GEARS-SGConv using scGPT/coexp perturbation vectors over gene-gene graph
    p_gears_s = run_gears_sgconv(xs_tr, Ys_tr, xs_ev, n_genes)
    np.save(os.path.join(OUT_DIR, "single_pred_GEARS_SGConv.npy"), p_gears_s)
    p_gears_c = run_gears_sgconv(xc_tr, Yc_tr, xc_ev, n_genes)
    np.save(os.path.join(OUT_DIR, "combo_pred_GEARS_SGConv.npy"), p_gears_c)

    # 3. Merge all SOTA open-source models + Exp37c / Exp39 into unified tables!
    df_37c = pd.read_csv(r"D:\data\AIVC\test_results\exp37c_overexpression_results.csv")
    df_39 = pd.read_csv(r"D:\data\AIVC\test_results\exp39_norman_combo_epistasis_results.csv")

    s_rows = df_37c.to_dict(orient="records")
    c_rows = df_39[["model", "mean_pearson", "top20_pearson", "dir_acc_top20", "mse", "epistasis_r"]].to_dict(orient="records")

    for tag, disp in [
        ("Geneformer_V2_104M", "Geneformer-V2 (104M)"),
        ("scFoundation_100M", "scFoundation (100M)"),
        ("scGPT", "scGPT (WholeHuman)"),
        ("GEARS_SGConv", "GEARS (SGConv)"),
    ]:
        ps = np.load(os.path.join(OUT_DIR, f"single_pred_{tag}.npy"))
        pc = np.load(os.path.join(OUT_DIR, f"combo_pred_{tag}.npy"))
        ms = eval_metrics(Ys_ev, ps)
        mc = eval_metrics(Yc_ev, pc, Xadd_c_ev)
        s_rows.append({"model": disp, **ms})
        c_rows.append({"model": disp, **mc})

    df_s_out = pd.DataFrame(s_rows).sort_values("top20_pearson", ascending=False).reset_index(drop=True)
    df_c_out = pd.DataFrame(c_rows).sort_values("top20_pearson", ascending=False).reset_index(drop=True)
    df_s_out.to_csv(OUT_SINGLE_CSV, index=False, encoding="utf-8-sig")
    df_c_out.to_csv(OUT_COMBO_CSV, index=False, encoding="utf-8-sig")

    print("\n=== Unified Norman2019 Unseen Single-Gene Overexpression Leaderboard ===")
    print(df_s_out.to_string(index=False))
    print("\n=== Unified Norman2019 Unseen Double-Gene Combo & Epistasis Leaderboard ===")
    print(df_c_out.to_string(index=False))
    print(f"\nCompleted Part 2 in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
