# -*- coding: utf-8 -*-
"""
Exp11b: 多基因组合中的“主效驱动靶点 (Driver) vs 次要跟随靶点 (Follower)”定量归因分析
基于 Tian2019_iPSC_combo_hvg2000.h5ad 与 AIVC-OmniFusion (Set-Attention) 模型，
通过 Leave-One-Out 边际效应分解，量化每个靶点在多点组合中的主导权重与冗余度。
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from omni_fusion import GatedMultiTeacherFusionNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def run_attribution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Tian2019_iPSC_combo_hvg2000.h5ad")
    obs = adata.obs.copy()
    X_all = _to_dense_array(adata.X).astype(np.float32)
    ctrl_mask = (obs["canon_pert"] == "control").values
    mu_ctrl = X_all[ctrl_mask].mean(axis=0)

    deltas, n_genes_map = {}, {}
    for cp, grp in obs[~ctrl_mask].groupby("canon_pert", observed=True):
        idx = obs.index.get_indexer(grp.index)
        deltas[str(cp)] = X_all[idx].mean(axis=0) - mu_ctrl
        n_genes_map[str(cp)] = int(grp["n_canon"].iloc[0])

    tr_keys = sorted([k for k, n in n_genes_map.items() if n in (1, 2)])
    tr_vocab = sorted({t for k in tr_keys for t in k.split("_")})
    tr_vocab_set = set(tr_vocab)
    te_keys = sorted([k for k, n in n_genes_map.items() if n >= 3 and all(t in tr_vocab_set for t in k.split("_"))])

    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(keys):
        m = np.zeros((len(keys), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(keys):
            for t in ps.split("_"):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True).fit(enc_lin(tr_keys), np.vstack([deltas[k] for k in tr_keys]))
    t2id = {t: i + 1 for i, t in enumerate(tr_vocab)}
    max_len = 6

    var_ens = {g: str(adata.var["ensemble_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensemble_id" in adata.var.columns else {}
    gf_dict = extract_geneformer_gene_embeddings(
        r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl",
        tr_vocab, var_ens,
    )
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", tr_vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", tr_vocab)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    coexp_dict = build_gene_coexp_prior(adata[obs["n_canon"] <= 2].X, list(adata.var_names), prior_dim=64)

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}
    teacher_matrices = []
    for td in teachers.values():
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(tr_vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

    def make_tids(keys):
        arr = np.zeros((len(keys), max_len), dtype=np.int64)
        for i, p in enumerate(keys):
            for j, t in enumerate(p.split("_")[:max_len]):
                arr[i, j] = t2id.get(t, 0)
        return arr

    tr_tids = torch.tensor(make_tids(tr_keys), device=device)
    tr_dl = torch.tensor(ridge.predict(enc_lin(tr_keys)).astype(np.float32), device=device)
    tr_y = torch.tensor(np.vstack([deltas[k] for k in tr_keys]).astype(np.float32), device=device)
    tr_xc = torch.tensor(np.tile(mu_ctrl, (len(tr_keys), 1)).astype(np.float32), device=device)

    torch.manual_seed(42)
    net = GatedMultiTeacherFusionNet(
        n_genes=adata.n_vars, n_tokens=len(tr_vocab), teacher_matrices=teacher_matrices,
        bottleneck_dim=64, hidden_dim=192, teacher_ctx_dim=1152, dropout=0.20,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    net.train()
    for ep in range(80):
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

    @torch.no_grad()
    def predict_batch(keys):
        tids = torch.tensor(make_tids(keys), device=device)
        dl = torch.tensor(ridge.predict(enc_lin(keys)).astype(np.float32), device=device)
        xc = torch.tensor(np.tile(mu_ctrl, (len(keys), 1)).astype(np.float32), device=device)
        pred, _, _ = net(xc, tids, dl)
        return pred.cpu().numpy()

    # Leave-One-Out 边际贡献归因
    full_preds = predict_batch(te_keys)
    gene_contribs = {g: [] for g in tr_vocab}
    example_rows = []

    for idx, combo in enumerate(te_keys):
        genes = combo.split("_")
        loo_keys = ["_".join([g for j, g in enumerate(genes) if j != i]) for i in range(len(genes))]
        loo_preds = predict_batch(loo_keys)
        marg = np.linalg.norm(full_preds[idx][None, :] - loo_preds, axis=1)
        total_m = float(np.sum(marg)) + 1e-8
        shares = marg / total_m
        for g, sh, mval in zip(genes, shares, marg):
            gene_contribs[g].append((float(sh), float(mval)))
        if len(genes) == 3 and len(example_rows) < 10:
            order = np.argsort(-shares)
            example_rows.append({
                "combo": combo,
                "driver_gene": genes[order[0]],
                "driver_share": round(float(shares[order[0]]), 4),
                "secondary_gene": genes[order[1]],
                "secondary_share": round(float(shares[order[1]]), 4),
                "follower_gene": genes[order[2]],
                "follower_share": round(float(shares[order[2]]), 4),
            })

    rank_rows = []
    for g, vals in gene_contribs.items():
        if vals:
            rank_rows.append({
                "target_gene": g,
                "n_combos_appeared": len(vals),
                "mean_dominance_share": float(np.mean([v[0] for v in vals])),
                "mean_marginal_l2": float(np.mean([v[1] for v in vals])),
                "role": "Master Driver" if np.mean([v[0] for v in vals]) >= 0.35 else ("Co-Driver" if np.mean([v[0] for v in vals]) >= 0.22 else "Follower / Secondary"),
            })
    rank_df = pd.DataFrame(rank_rows).sort_values("mean_dominance_share", ascending=False)
    rank_df.to_csv(r"D:\data\AIVC\test_results\exp11b_target_driver_follower_ranking.csv", index=False, encoding="utf-8")
    pd.DataFrame(example_rows).to_csv(r"D:\data\AIVC\test_results\exp11b_3gene_decomposition_examples.csv", index=False, encoding="utf-8")

    print("\n=== 多基因组合中靶点主导度排名 (Top Master Drivers vs Followers) ===")
    print(rank_df.head(12).round(4).to_string(index=False))
    print("\n=== 3-Gene 组合主次拆解示例 ===")
    print(pd.DataFrame(example_rows).head(6).to_string(index=False))


if __name__ == "__main__":
    run_attribution()
