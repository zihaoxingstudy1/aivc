# -*- coding: utf-8 -*-
"""
sota_bench_part3b_scvi_vae.py
Fast pure-PyTorch scVI-style VAE (n_latent=32, n_hidden=128) + perturbation decoder
on rpe1_ood_bundle.npz (avoids Windows PyTorch-Lightning multiprocess deadlock).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"D:\data\AIVC\test_results\sota_bench_preds"


class ScVIVAELike(nn.Module):
    def __init__(self, n_genes: int, n_hidden: int = 128, n_latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU())
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)
        self.dec = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_genes))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, lv


def main():
    t0 = time.time()
    bundle = np.load(os.path.join(OUT_DIR, "rpe1_ood_bundle.npz"), allow_pickle=True)
    ctrl_names = [str(c) for c in bundle["ctrl_names"]]
    ctrl_map = {c: bundle["ctrl_vecs"][i] for i, c in enumerate(ctrl_names)}
    tr_lineages = [str(ln) for ln in bundle["tr_lineages"]]
    n_genes = len(bundle["target_genes"])

    X_tr_expr = np.vstack([np.clip(ctrl_map[ln] + bundle["Y_tr"][i], 0.0, None) for i, ln in enumerate(tr_lineages)]).astype(np.float32)
    X_vae = np.vstack([bundle["ctrl_vecs"], X_tr_expr]).astype(np.float32)

    vae = ScVIVAELike(n_genes, n_hidden=128, n_latent=32).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(X_vae)), batch_size=256, shuffle=True)
    vae.train()
    for _ in range(25):
        for (bx,) in dl:
            bx = bx.to(DEVICE)
            opt.zero_grad()
            rec, mu, lv = vae(bx)
            kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = F.mse_loss(rec, bx) + 0.01 * kld
            loss.backward()
            opt.step()
    vae.eval()
    with torch.no_grad():
        z_ctrls, _ = vae.encode(torch.tensor(bundle["ctrl_vecs"], device=DEVICE))
        z_ctrls = z_ctrls.cpu().numpy().astype(np.float32)
    z_map = {c: z_ctrls[i] for i, c in enumerate(ctrl_names)}

    X_tr = np.hstack([np.vstack([z_map[ln] for ln in tr_lineages]), bundle["X_coexp_tr"]]).astype(np.float32)
    X_ev = np.hstack([np.repeat(z_map["RPE1"][None, :], len(bundle["ev_keys"]), axis=0), bundle["X_coexp_ev"]]).astype(np.float32)

    dec = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(256, n_genes)
    ).to(DEVICE)
    opt2 = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-3)
    dl2 = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(bundle["Y_tr"])), batch_size=256, shuffle=True)
    dec.train()
    for _ in range(30):
        for bx, by in dl2:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt2.zero_grad()
            F.mse_loss(dec(bx), by).backward()
            opt2.step()
    dec.eval()
    with torch.no_grad():
        pred = dec(torch.tensor(X_ev, device=DEVICE)).cpu().numpy().astype(np.float32)
    np.save(os.path.join(OUT_DIR, "pred_scVI.npy"), pred)
    print(f"Saved pred_scVI.npy in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
