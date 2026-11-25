# -*- coding: utf-8 -*-
"""
不确定性量化 (MC-Dropout UQ) 与虚拟全组合协同筛选模块 (Uncertainty & Virtual Screening)
1. 评估模型预测不确定性与真实误差之间的校准关系 (Active Learning / 干湿闭环依据)；
2. 对未测过的基因组合执行大规模虚拟筛选，发现高协同、高置信度的潜在新靶点组合。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omni_fusion import GatedMultiTeacherFusionNet
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from lbb import _compute_group_deltas, _to_dense_array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_python_scripts"))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def run_uq_and_virtual_screening(
    split_h5ad: str,
    ckpt_path: str,
    out_uq_csv: str,
    out_uq_png: str,
    out_vs_csv: str,
    mc_samples: int = 30,
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

    all_p = sorted(set(tr_d) | set(ev_d) | set(adata.obs.loc[adata.obs["perturbation"] != "control", "perturbation"].astype(str)))
    vocab = sorted({t.strip() for p in all_p for t in p.split("_") if t.strip()})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = 2

    # 加载 5 源先验
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
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_lin(tr_keys), np.vstack([tr_d[p] for p in tr_keys]))

    net = GatedMultiTeacherFusionNet(
        n_genes=adata.n_vars,
        n_tokens=len(vocab),
        teacher_matrices=teacher_matrices,
        bottleneck_dim=64,
        hidden_dim=192,
        teacher_ctx_dim=1152,
        dropout=0.20,
    ).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # 1. MC-Dropout 不确定性评估 (在测试集上开启 dropout 采样 mc_samples 次)
    ev_keys = sorted(ev_d.keys())
    ev_tids = np.zeros((len(ev_keys), max_len), dtype=np.int64)
    for i, p in enumerate(ev_keys):
        for j, t in enumerate(p.split("_")[:max_len]):
            ev_tids[i, j] = t2id.get(t.strip(), 0)
    ev_dl = torch.tensor(ridge.predict(enc_lin(ev_keys)).astype(np.float32), device=device)
    ev_xc = torch.tensor(np.tile(mu_ev, (len(ev_keys), 1)).astype(np.float32), device=device)
    ev_tid_t = torch.tensor(ev_tids, device=device)

    net.train()  # 开启 MC-Dropout
    preds_mc = []
    with torch.no_grad():
        for _ in range(mc_samples):
            p_mc, _, _ = net(ev_xc, ev_tid_t, ev_dl)
            preds_mc.append(p_mc.cpu().numpy())
    preds_mc = np.stack(preds_mc, axis=0)  # [T, N, G]
    pred_mean = preds_mc.mean(axis=0)
    pred_std = preds_mc.std(axis=0)  # [N, G]

    uq_rows = []
    for i, p in enumerate(ev_keys):
        dt = ev_d[p]
        dp = pred_mean[i]
        mse = float(np.mean((dt - dp) ** 2))
        r, _ = pearsonr(dt, dp)
        unc = float(np.mean(pred_std[i]))
        uq_rows.append({"perturbation": p, "uncertainty": unc, "mse_all": mse, "pearson_delta_all": float(r)})

    uq_df = pd.DataFrame(uq_rows)
    sp_r, sp_p = spearmanr(uq_df["uncertainty"], uq_df["mse_all"])
    print(f"[UQ] MC-Dropout 不确定性与真实 MSE 误差 Spearman 相关系数: r = {sp_r:.4f} (p = {sp_p:.4g})")
    uq_df.to_csv(out_uq_csv, index=False, encoding="utf-8")

    # 绘制 UQ Calibration 图
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=180)
    sns.regplot(data=uq_df, x="uncertainty", y="mse_all", ax=ax, scatter_kws={"s": 60, "alpha": 0.8, "color": "#2980b9"}, line_kws={"color": "#e74c3c"})
    ax.set_title(f"MC-Dropout Uncertainty vs Prediction MSE (Spearman r={sp_r:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Model Epistemic Uncertainty (Mean Std)", fontsize=10)
    ax.set_ylabel("True Prediction MSE", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_uq_png, bbox_inches="tight")
    plt.close(fig)

    # 2. 大规模虚拟组合筛选 (从未实验过的单基因两两组合中筛选 1000 对，寻找强非线性协同且高置信度的候选靶点)
    observed_combos = set(all_p)
    singles = sorted([t for t in tr_vocab if t in t2id])
    cand_pairs = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            c1 = f"{singles[i]}_{singles[j]}"
            c2 = f"{singles[j]}_{singles[i]}"
            if c1 not in observed_combos and c2 not in observed_combos:
                cand_pairs.append(c1)

    rng = np.random.default_rng(42)
    if len(cand_pairs) > 1000:
        cand_pairs = list(rng.choice(cand_pairs, size=1000, replace=False))

    c_tids = np.zeros((len(cand_pairs), max_len), dtype=np.int64)
    for i, p in enumerate(cand_pairs):
        for j, t in enumerate(p.split("_")[:max_len]):
            c_tids[i, j] = t2id.get(t.strip(), 0)
    c_dl = ridge.predict(enc_lin(cand_pairs)).astype(np.float32)
    c_dl_t = torch.tensor(c_dl, device=device)
    c_xc_t = torch.tensor(np.tile(mu_ev, (len(cand_pairs), 1)).astype(np.float32), device=device)
    c_tid_t = torch.tensor(c_tids, device=device)

    net.eval()
    with torch.no_grad():
        c_pred, _, _ = net(c_xc_t, c_tid_t, c_dl_t)
        c_pred_np = c_pred.cpu().numpy()

    # 计算协同效应强度 (Synergy = ||Delta_omni - Delta_ridge||_2) 与总转录组重编程幅度
    synergy = np.linalg.norm(c_pred_np - c_dl, axis=1)
    mag = np.linalg.norm(c_pred_np, axis=1)
    vs_df = pd.DataFrame({
        "virtual_combo": cand_pairs,
        "predicted_reprogramming_magnitude": mag,
        "epistemic_synergy_score": synergy,
    }).sort_values("epistemic_synergy_score", ascending=False)
    vs_df.to_csv(out_vs_csv, index=False, encoding="utf-8")
    print("\n=== 虚拟筛选 Top 10 未实验高协同双靶点组合 ===")
    print(vs_df.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    run_uq_and_virtual_screening(
        split_h5ad=r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad",
        ckpt_path=r"D:\data\AIVC\test_results\norman2019_omnifusion_v2.pt",
        out_uq_csv=r"D:\data\AIVC\test_results\norman2019_uq_calibration.csv",
        out_uq_png=r"D:\data\AIVC\test_results\norman2019_uq_calibration.png",
        out_vs_csv=r"D:\data\AIVC\test_results\norman2019_virtual_screening_top_combos.csv",
        mc_samples=30,
    )
