# -*- coding: utf-8 -*-
"""
方向 4 升级 (Exp6): Deep Ensemble (5-Seed Epistemic Variance) + 5-Teacher Semantic OOD Distance UQ
解决 MC-Dropout (r=-0.055) 无法有效校准预测误差的问题，验证综合不确定性与真实 MSE 之间的正相关性。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from omni_fusion import GatedMultiTeacherFusionNet
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from lbb import _compute_group_deltas, _to_dense_array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def run_deep_ensemble_uq(
    split_h5ad: str,
    out_uq_csv: str,
    out_uq_png: str,
    seeds: tuple = (42, 101, 202, 303, 404),
    epochs: int = 80,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(split_h5ad)
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    def get_cm(sub):
        m = (sub.obs["perturbation"].astype(str) == "control").values
        return _to_dense_array(sub.X[m]).mean(axis=0)

    mu_tr, mu_ev = get_cm(tr), get_cm(ev)
    tr_d = _compute_group_deltas(tr[tr.obs["perturbation"] != "control"], "perturbation", mu_tr)
    ev_d = _compute_group_deltas(ev[ev.obs["perturbation"] != "control"], "perturbation", mu_ev)

    all_p = sorted(set(tr_d) | set(ev_d))
    vocab = sorted({t.strip() for p in all_p for t in p.split("_") if t.strip()})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = 2

    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"
    var_ens = {g: str(adata.var["ensemble_id"].iloc[i]) for i, g in enumerate(adata.var_names)}

    gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    coexp_dict = build_gene_coexp_prior(tr.X, list(adata.var_names), prior_dim=64)

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}
    teacher_matrices = []
    for td in teachers.values():
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

    tr_vocab = sorted({t for p in tr_d for t in p.split("_") if t.strip()})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(plist):
            for t in ps.split("_"):
                if t.strip() in lin_idx:
                    m[ri, lin_idx[t.strip()]] = 1.0
        return m

    tr_keys = sorted(tr_d.keys())
    ev_keys = sorted(ev_d.keys())
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_lin(tr_keys), np.vstack([tr_d[p] for p in tr_keys]))

    def make_tids(keys):
        arr = np.zeros((len(keys), max_len), dtype=np.int64)
        for i, p in enumerate(keys):
            for j, t in enumerate(p.split("_")[:max_len]):
                arr[i, j] = t2id.get(t.strip(), 0)
        return arr

    tr_tids = torch.tensor(make_tids(tr_keys), device=device)
    tr_dl = torch.tensor(ridge.predict(enc_lin(tr_keys)).astype(np.float32), device=device)
    tr_y = torch.tensor(np.vstack([tr_d[p] for p in tr_keys]).astype(np.float32), device=device)
    tr_xc = torch.tensor(np.tile(mu_tr, (len(tr_keys), 1)).astype(np.float32), device=device)

    ev_tids = torch.tensor(make_tids(ev_keys), device=device)
    ev_dl = torch.tensor(ridge.predict(enc_lin(ev_keys)).astype(np.float32), device=device)
    ev_xc = torch.tensor(np.tile(mu_ev, (len(ev_keys), 1)).astype(np.float32), device=device)

    # 1. 计算 5-Teacher 拼接语义空间中测试扰动到最近训练扰动的 Cosine OOD Distance
    concat_mats = []
    for mat in teacher_matrices:
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm = np.where(norm > 1e-8, norm, 1.0)
        concat_mats.append(mat / norm)
    sem_vocab = np.hstack(concat_mats)  # [V+1, sum_D]

    def pert_sem_vec(keys):
        vecs = []
        for p in keys:
            toks = [t2id.get(t.strip(), 0) for t in p.split("_")[:max_len] if t.strip()]
            v = sem_vocab[toks].mean(axis=0) if toks else np.zeros(sem_vocab.shape[1], dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.vstack(vecs)

    tr_sem = pert_sem_vec(tr_keys)
    ev_sem = pert_sem_vec(ev_keys)
    sim_mat = ev_sem @ tr_sem.T
    ood_dist = 1.0 - sim_mat.max(axis=1)  # 1 - max_cos_sim

    # 2. 训练 5-Seed Deep Ensemble
    ensemble_preds = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        net = GatedMultiTeacherFusionNet(
            n_genes=adata.n_vars,
            n_tokens=len(vocab),
            teacher_matrices=teacher_matrices,
            bottleneck_dim=64,
            hidden_dim=192,
            teacher_ctx_dim=1152,
            dropout=0.20,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        net.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr_keys), device=device)
            for i in range(0, len(tr_keys), 64):
                idx = perm[i : i + 64]
                opt.zero_grad()
                pred, _, _ = net(tr_xc[idx], tr_tids[idx], tr_dl[idx])
                loss = F.mse_loss(pred, tr_y[idx]) + 0.1 * (1.0 - F.cosine_similarity(pred, tr_y[idx], dim=-1).mean())
                loss.backward()
                opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            ev_pred, _, _ = net(ev_xc, ev_tids, ev_dl)
            ensemble_preds.append(ev_pred.cpu().numpy())

    ens_stack = np.stack(ensemble_preds, axis=0)  # [S, N, G]
    ens_mean = ens_stack.mean(axis=0)
    ens_var = ens_stack.var(axis=0).mean(axis=1)  # [N] Epistemic Variance
    ens_std = ens_stack.std(axis=0).mean(axis=1)
    pred_norm = np.linalg.norm(ens_mean, axis=1)  # 预测响应幅度 (高响应幅度往往伴随更大绝对MSE)

    # 归一化融合指标: Z(ens_std) + Z(ood_dist)
    z_std = (ens_std - ens_std.mean()) / (ens_std.std() + 1e-8)
    z_ood = (ood_dist - ood_dist.mean()) / (ood_dist.std() + 1e-8)
    combined_unc = 0.7 * z_std + 0.3 * z_ood

    rows = []
    for i, p in enumerate(ev_keys):
        dt = ev_d[p]
        dp = ens_mean[i]
        mse = float(np.mean((dt - dp) ** 2))
        r, _ = pearsonr(dt, dp)
        rows.append({
            "perturbation": p,
            "ensemble_std": float(ens_std[i]),
            "ensemble_var": float(ens_var[i]),
            "semantic_ood_dist": float(ood_dist[i]),
            "combined_uncertainty": float(combined_unc[i]),
            "pred_magnitude": float(pred_norm[i]),
            "mse_all": mse,
            "pearson_delta_all": float(r),
        })

    df = pd.DataFrame(rows)
    r_std, p_std = spearmanr(df["ensemble_std"], df["mse_all"])
    r_ood, p_ood = spearmanr(df["semantic_ood_dist"], df["mse_all"])
    r_comb, p_comb = spearmanr(df["combined_uncertainty"], df["mse_all"])
    print(f"[DeepEnsemble-UQ] 5-Seed Std vs MSE: Spearman r={r_std:.4f} (p={p_std:.4g})")
    print(f"[DeepEnsemble-UQ] Semantic OOD Dist vs MSE: Spearman r={r_ood:.4f} (p={p_ood:.4g})")
    print(f"[DeepEnsemble-UQ] Combined (Std+OOD) vs MSE: Spearman r={r_comb:.4f} (p={p_comb:.4g})")
    print(f"[DeepEnsemble-UQ] 5-Seed Ensemble Mean Pearson: {df['pearson_delta_all'].mean():.4f} | Mean MSE: {df['mse_all'].mean():.6f}")
    df.to_csv(out_uq_csv, index=False, encoding="utf-8")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    sns.regplot(data=df, x="ensemble_std", y="mse_all", ax=axes[0], scatter_kws={"s": 55, "alpha": 0.8, "color": "#2980b9"}, line_kws={"color": "#e74c3c"})
    axes[0].set_title(f"5-Seed Deep Ensemble Std vs MSE\nSpearman r = {r_std:.3f} (p = {p_std:.3g})", fontweight="bold")
    axes[0].set_xlabel("5-Seed Epistemic Std")
    axes[0].set_ylabel("True Prediction MSE")

    sns.regplot(data=df, x="combined_uncertainty", y="mse_all", ax=axes[1], scatter_kws={"s": 55, "alpha": 0.8, "color": "#27ae60"}, line_kws={"color": "#e74c3c"})
    axes[1].set_title(f"Combined (Ensemble Std + Semantic OOD) vs MSE\nSpearman r = {r_comb:.3f} (p = {p_comb:.3g})", fontweight="bold")
    axes[1].set_xlabel("Combined Epistemic + Semantic OOD Uncertainty (Z-score)")
    axes[1].set_ylabel("True Prediction MSE")

    plt.tight_layout()
    plt.savefig(out_uq_png, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_deep_ensemble_uq(
        split_h5ad=r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad",
        out_uq_csv=r"D:\data\AIVC\test_results\exp6_deep_ensemble_uq.csv",
        out_uq_png=r"D:\data\AIVC\test_results\exp6_deep_ensemble_uq.png",
    )
