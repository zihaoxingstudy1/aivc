# -*- coding: utf-8 -*-
"""
sota_bench_part3_gears_scgpt_scvi.py
Run in bench_sota env (torch 2.3.0+cu121, scgpt 0.2.4, scvi-tools 1.1.6, gears 0.1.2, torch_geometric 2.8.1):
  1. scGPT-WholeHuman (512-dim pretrained TransformerModel + perturbation adapter)
  2. scVI (scvi-tools SCVI VAE latent + perturbation decoder)
  3. GEARS-SGConv (torch_geometric SGConv co-expression + perturbation GNN)
on the exact frozen rpe1_ood_bundle.npz (7-Lineage -> Unseen RPE1, 2,016 perturbations).
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import load_file as load_safetensors
import scvi
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from torch_geometric.nn import SGConv

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"
SCGPT_DIR = r"D:\data\AIVC\pretrained\scgpt"


def run_scgpt(bundle: dict) -> np.ndarray:
    print("\n--- [SOTA 1/3] scGPT-WholeHuman (512d Transformer) ---")
    vocab = GeneVocab.from_file(os.path.join(SCGPT_DIR, "vocab.json"))
    with open(os.path.join(SCGPT_DIR, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pad_token = "<pad>"
    if pad_token not in vocab:
        pad_token = vocab.itos[0]
    pad_id = vocab[pad_token]

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(cfg.get("embsize", 512)),
        nhead=int(cfg.get("nhead", 8)),
        d_hid=int(cfg.get("d_hid", 512)),
        nlayers=int(cfg.get("nlayers", 12)),
        vocab=vocab,
        dropout=0.0,
        pad_token=pad_token,
        pad_value=int(cfg.get("pad_value", -2)),
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        input_emb_style="continuous",
        cell_emb_style="cls",
        use_fast_transformer=False,
    )
    sd = load_safetensors(os.path.join(SCGPT_DIR, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded scGPT weights: missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(DEVICE)
    model.eval()

    gene_emb_mat = model.encoder.embedding.weight.detach().cpu().numpy()  # (vocab, 512)
    target_genes = [str(g) for g in bundle["target_genes"]]
    ctrl_names = [str(c) for c in bundle["ctrl_names"]]
    ctrl_vecs = bundle["ctrl_vecs"]

    # Map genes to scGPT vocab
    valid_idx = []
    valid_toks = []
    for i, g in enumerate(target_genes):
        if g in vocab:
            valid_idx.append(i)
            valid_toks.append(vocab[g])
        elif g.upper() in vocab:
            valid_idx.append(i)
            valid_toks.append(vocab[g.upper()])
    print(f"  scGPT matched {len(valid_idx)}/{len(target_genes)} target genes")

    cls_id = vocab["<cls>"] if "<cls>" in vocab else pad_id
    cell_embs = {}
    with torch.no_grad():
        for cname, cvec in zip(ctrl_names, ctrl_vecs):
            vals = cvec[valid_idx]
            # Bin into 51 bins or pass continuous non-zero top 1200 genes
            nz = np.where(vals > 0)[0]
            if len(nz) > 1200:
                nz = nz[np.argsort(-vals[nz])[:1200]]
            toks = np.array([cls_id] + [valid_toks[j] for j in nz], dtype=np.int64)
            exps = np.array([0.0] + [float(vals[j]) for j in nz], dtype=np.float32)
            src = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
            val = torch.tensor(exps, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            pad_mask = src.eq(pad_id)
            out = model._encode(src, val, src_key_padding_mask=pad_mask)
            cell_embs[cname] = out[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    del model
    torch.cuda.empty_cache()

    def get_gene_emb(p):
        if p in vocab:
            return gene_emb_mat[vocab[p]].astype(np.float32)
        if str(p).upper() in vocab:
            return gene_emb_mat[vocab[str(p).upper()]].astype(np.float32)
        return np.zeros(gene_emb_mat.shape[1], dtype=np.float32)

    X_tr = np.hstack([
        np.vstack([cell_embs[str(ln)] for ln in bundle["tr_lineages"]]),
        np.vstack([get_gene_emb(p) for p in bundle["tr_perts"]]),
    ]).astype(np.float32)
    X_ev = np.hstack([
        np.repeat(cell_embs["RPE1"][None, :], len(bundle["ev_keys"]), axis=0),
        np.vstack([get_gene_emb(p) for p in bundle["ev_keys"]]),
    ]).astype(np.float32)

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
        return adapter(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)


def run_scvi(bundle: dict) -> np.ndarray:
    print("\n--- [SOTA 2/3] scVI (scvi-tools VAE + Perturbation Decoder) ---")
    ctrl_names = [str(c) for c in bundle["ctrl_names"]]
    ctrl_map = {c: bundle["ctrl_vecs"][i] for i, c in enumerate(ctrl_names)}
    # Build pseudo-count matrix from training perturbed states + RPE1 control
    tr_lineages = [str(ln) for ln in bundle["tr_lineages"]]
    X_tr_expr = np.vstack([np.clip(ctrl_map[ln] + bundle["Y_tr"][i], 0.0, None) for i, ln in enumerate(tr_lineages)])
    # Subsample 2500 rows + all 8 lineage controls for fast SCVI fit
    rng = np.random.RandomState(SEED)
    sub_idx = rng.choice(len(X_tr_expr), size=min(2500, len(X_tr_expr)), replace=False)
    X_scvi = np.vstack([bundle["ctrl_vecs"], X_tr_expr[sub_idx]])
    counts = np.round(np.expm1(np.clip(X_scvi, 0.0, 8.0)) * 5.0).astype(np.float32)
    adata_scvi = ad.AnnData(X=counts)
    scvi.model.SCVI.setup_anndata(adata_scvi)
    vae = scvi.model.SCVI(adata_scvi, n_latent=32, n_hidden=128, n_layers=2)
    vae.train(
        max_epochs=25,
        batch_size=256,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False,
        datasplitter_kwargs={"num_workers": 0},
    )
    z_ctrls = vae.get_latent_representation(ad.AnnData(X=np.round(np.expm1(np.clip(bundle["ctrl_vecs"], 0.0, 8.0)) * 5.0).astype(np.float32)))
    z_map = {c: z_ctrls[i].astype(np.float32) for i, c in enumerate(ctrl_names)}

    X_tr = np.hstack([
        np.vstack([z_map[ln] for ln in tr_lineages]),
        bundle["X_coexp_tr"],
    ]).astype(np.float32)
    X_ev = np.hstack([
        np.repeat(z_map["RPE1"][None, :], len(bundle["ev_keys"]), axis=0),
        bundle["X_coexp_ev"],
    ]).astype(np.float32)

    torch.manual_seed(SEED)
    dec = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(256, len(bundle["target_genes"]))
    ).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(bundle["Y_tr"])), batch_size=256, shuffle=True)
    dec.train()
    for _ in range(30):
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(dec(bx), by).backward()
            opt.step()
    dec.eval()
    with torch.no_grad():
        return dec(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)


class GEARSSGConvNet(nn.Module):
    """GEARS-style SGConv gene-graph propagation over perturbation embeddings."""
    def __init__(self, n_genes: int, d_pert: int, edge_index: torch.Tensor, hid: int = 64):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.gene_base = nn.Parameter(torch.randn(n_genes, hid) * 0.05)
        self.pert_proj = nn.Linear(d_pert, hid)
        self.sgc = SGConv(hid, hid, K=2)
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(hid, 1))

    def forward(self, pert_feat: torch.Tensor) -> torch.Tensor:
        # pert_feat: (B, d_pert) -> broadcast add to gene graph nodes and run SGConv
        B = pert_feat.shape[0]
        p_emb = self.pert_proj(pert_feat)  # (B, hid)
        # Efficient batched SGConv via linear combination of base + pert_proj
        h_base = self.sgc(self.gene_base, self.edge_index)  # (G, hid)
        h = h_base.unsqueeze(0) + p_emb.unsqueeze(1)  # (B, G, hid)
        return self.readout(h).squeeze(-1)


def run_gears_sgconv(bundle: dict) -> np.ndarray:
    print("\n--- [SOTA 3/3] GEARS-SGConv (Gene Co-expression Graph + PyG SGConv) ---")
    Y_tr = bundle["Y_tr"]
    n_genes = len(bundle["target_genes"])
    # Build top-k=15 gene-gene co-expression graph from Y_tr (strictly train only)
    Y_norm = Y_tr - Y_tr.mean(axis=0, keepdims=True)
    Y_norm = Y_norm / (Y_norm.std(axis=0, keepdims=True) + 1e-6)
    corr = (Y_norm.T @ Y_norm) / float(len(Y_tr))
    np.fill_diagonal(corr, 0.0)
    src_l, dst_l = [], []
    for i in range(n_genes):
        nbrs = np.argsort(-np.abs(corr[i]))[:15]
        for j in nbrs:
            src_l.append(i)
            dst_l.append(int(j))
    edge_index = torch.tensor([src_l, dst_l], dtype=torch.long, device=DEVICE)

    X_tr = bundle["X_coexp_tr"].astype(np.float32)
    X_ev = bundle["X_coexp_ev"].astype(np.float32)

    torch.manual_seed(SEED)
    model = GEARSSGConvNet(n_genes, X_tr.shape[1], edge_index, hid=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), batch_size=256, shuffle=True)
    model.train()
    for _ in range(30):
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(bx), by).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = []
        ev_t = torch.tensor(X_ev, device=DEVICE)
        for s0 in range(0, len(ev_t), 256):
            preds.append(model(ev_t[s0:s0 + 256]).cpu().numpy())
        return np.vstack(preds).astype(np.float32)


def main():
    t0 = time.time()
    print(f"=== SOTA Benchmark Part 3: scGPT + scVI + GEARS-SGConv ({DEVICE}) ===")
    bundle = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)

    scgpt_fp = os.path.join(OUT_DIR, "pred_scGPT.npy")
    if not os.path.exists(scgpt_fp):
        p_scgpt = run_scgpt(bundle)
        np.save(scgpt_fp, p_scgpt)
    else:
        print("  [SKIP] pred_scGPT.npy already exists")

    p_gears = run_gears_sgconv(bundle)
    np.save(os.path.join(OUT_DIR, "pred_GEARS_SGConv.npy"), p_gears)

    p_scvi = run_scvi(bundle)
    np.save(os.path.join(OUT_DIR, "pred_scVI.npy"), p_scvi)

    print(f"Part 3 completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
