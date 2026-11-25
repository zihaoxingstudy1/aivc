import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from bfc import build_gene_coexp_prior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings, load_scgpt_gene_embeddings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_CSV = r"D:\data\AIVC\test_results\exp29b_boost_rpe1_results.csv"


class PerProgramGatedCrossLineageNet(nn.Module):
    """
    Upgraded Exp29b:
      - K=128 Gene-Program SVD basis
      - Per-program modulation (predicts both residual programs & per-program scaling on Ridge prior)
      - Stronger Cosine Direction Loss (0.25) + Coexp/ESM2/scGPT Tri-Teacher Embeddings
    """
    def __init__(self, n_genes: int, n_teacher: int, U_prog: np.ndarray, hid: int = 192):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))  # (K, G)
        k_prog = U_prog.shape[0]

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.prog_gate = nn.Sequential(nn.Linear(64 + 3, 64), nn.GELU(), nn.Linear(64, k_prog))
        self.prog_res = nn.Sequential(
            nn.Linear(64 + hid + 3, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.prog_gate[-1].weight)
        nn.init.zeros_(self.prog_gate[-1].bias)
        nn.init.zeros_(self.prog_res[-1].weight)
        nn.init.zeros_(self.prog_res[-1].bias)

    def forward(self, x_ctrl, x_teach, x_ridge, x_tstat):
        hc = self.ctrl_enc(x_ctrl)
        # Project Ridge prior onto K=128 gene programs and modulate per-program by target cell state!
        ridge_prog = x_ridge @ self.U_prog.T  # (B, K)
        gate_k = 1.0 + 0.5 * torch.tanh(self.prog_gate(torch.cat([hc, x_tstat], dim=-1)))
        res_k = self.prog_res(torch.cat([hc, self.teach_enc(x_teach), x_tstat], dim=-1))
        # Preserve orthogonal Ridge detail + modulated program signal
        ridge_orth = x_ridge - (ridge_prog @ self.U_prog)
        return ridge_orth + ((gate_k * ridge_prog + 0.5 * res_k) @ self.U_prog)


def main():
    print(f"=== Exp29b: Boosting All-Gene Pearson on Unseen RPE1 ({DEVICE}) ===")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad")
    target_genes = list(ref.var_names)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}

    tr_ref = ref[ref.obs["ood_split"] == "train"]
    ev_ref = ref[ref.obs["ood_split"] == "test"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    mu_rpe1 = _to_dense_array(ev_ref[ev_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    d_rpe1 = _compute_group_deltas(ev_ref[ev_ref.obs["perturbation"] != "control"], "perturbation", mu_rpe1)

    coexp_dict = build_gene_coexp_prior(tr_ref.X, target_genes, prior_dim=64)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    del ref, tr_ref, ev_ref

    lineage_specs = [("K562", mu_k562, d_k562)]
    for lname, fpath, sp_only in [
        ("HepG2", r"D:\data\AIVC\data\NadigOConner2024_hepg2.h5ad", False),
        ("Jurkat", r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad", False),
        ("iPSC", r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad", True),
        ("Norman", r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad", True),
        ("PrimaryT", r"D:\data\AIVC\data\ShifrutMarson2018.h5ad", True),
        ("THP1", r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad", True),
    ]:
        if os.path.exists(fpath):
            mc, dd = extract_lineage_deltas(fpath, target_genes, lname, min_cells=10, single_pert_only=sp_only)
            lineage_specs.append((lname, mc, dd))

    tr_samples = [(ln, p, mc, dv) for ln, mc, dd in lineage_specs for p, dv in dd.items() if "_" not in p and "+" not in p]
    ev_keys = sorted(d_rpe1.keys())
    tr_vocab = sorted({s[1] for s in tr_samples})
    all_vocab = sorted(set(tr_vocab) | set(ev_keys))
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", all_vocab)

    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    X_tr_lin = enc_lin([s[1] for s in tr_samples])
    Y_tr_lin = np.vstack([s[3] for s in tr_samples])
    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_lin)
    pred_ridge_ev = ridge.predict(enc_lin(ev_keys))

    svd = TruncatedSVD(n_components=128, random_state=SEED).fit(Y_tr_lin)
    U_prog = svd.components_.astype(np.float32)

    esm_dim = len(next(iter(esm2_dict.values())))
    co_dim = len(next(iter(coexp_dict.values())))
    sc_dim = len(next(iter(scgpt_dict.values())))

    def get_teach_vec(p):
        return np.concatenate([
            coexp_dict.get(p, np.zeros(co_dim, dtype=np.float32)),
            esm2_dict.get(p, np.zeros(esm_dim, dtype=np.float32)),
            scgpt_dict.get(p, np.zeros(sc_dim, dtype=np.float32)),
        ]).astype(np.float32)

    def get_target_stats(mc, p, r_vec):
        mu_m, sd_m = float(mc.mean()), max(float(mc.std()), 1e-6)
        raw_e = float(mc[gene_to_idx[p]]) if p in gene_to_idx else mu_m
        return np.array([raw_e, (raw_e - mu_m) / sd_m, float(np.log1p(np.linalg.norm(r_vec)))], dtype=np.float32)

    pred_ridge_tr = np.zeros_like(Y_tr_lin)
    lineages_tr = np.array([s[0] for s in tr_samples])
    for ln in np.unique(lineages_tr):
        m_out = (lineages_tr == ln)
        r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[~m_out], Y_tr_lin[~m_out])
        pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out])

    X_tr_ctrl = np.vstack([s[2] for s in tr_samples]).astype(np.float32)
    X_tr_teach = np.vstack([get_teach_vec(s[1]) for s in tr_samples])
    X_tr_tstat = np.vstack([get_target_stats(s[2], s[1], pred_ridge_tr[i]) for i, s in enumerate(tr_samples)])
    X_ev_ctrl = np.tile(mu_rpe1, (len(ev_keys), 1)).astype(np.float32)
    X_ev_teach = np.vstack([get_teach_vec(k) for k in ev_keys])
    X_ev_tstat = np.vstack([get_target_stats(mu_rpe1, k, pred_ridge_ev[i]) for i, k in enumerate(ev_keys)])

    net = PerProgramGatedCrossLineageNet(len(target_genes), X_tr_teach.shape[1], U_prog).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=2e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr_ctrl), torch.tensor(X_tr_teach), torch.tensor(pred_ridge_tr), torch.tensor(X_tr_tstat), torch.tensor(Y_tr_lin)), batch_size=256, shuffle=True)
    net.train()
    for _ in range(50):
        for bc, bt, br, bs, by in loader:
            bc, bt, br, bs, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bs.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            pred = net(bc, bt, br, bs)
            loss = F.mse_loss(pred, by) + 0.08 * (1.0 - F.cosine_similarity(pred, by, dim=-1)).mean()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred_29b = net(
            torch.tensor(X_ev_ctrl, device=DEVICE), torch.tensor(X_ev_teach, device=DEVICE),
            torch.tensor(pred_ridge_ev, device=DEVICE), torch.tensor(X_ev_tstat, device=DEVICE)
        ).cpu().numpy()

    df_29b = evaluate_perturbation_predictions(d_rpe1, {k: pred_29b[i] for i, k in enumerate(ev_keys)}, "Exp29b_PerProgramGated")
    out_row = pd.DataFrame([{
        "model": "Exp29b_PerProgramGated",
        "mean_pearson": df_29b["pearson_delta_all"].mean(),
        "top20_pearson": df_29b["pearson_delta_top20"].mean(),
        "dir_acc_top20": df_29b["dir_acc_top20"].mean(),
        "mse": df_29b["mse_all"].mean(),
    }])
    out_row.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n=== Exp29b Result on Unseen RPE1 (2,016 perturbations) ===")
    print(out_row.to_string(index=False))


if __name__ == "__main__":
    main()
