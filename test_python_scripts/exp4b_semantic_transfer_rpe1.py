# -*- coding: utf-8 -*-
"""
弱项 1 精准破局 (Exp4B):
基于三教师联合语义相似性核迁移 + 靶基因表达缩放，专门攻克 Replogle RPE1 的 340 个未见基因与 1676 个共有基因！
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_scfoundation_gene_embeddings


if __name__ == "__main__":
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad")
    tr = adata[adata.obs["ood_split"] == "train"]
    ev = adata[adata.obs["ood_split"] == "test"]

    tr_cm = (tr.obs["perturbation"].astype(str) == "control").values
    ev_cm = (ev.obs["perturbation"].astype(str) == "control").values
    mu_k562 = _to_dense_array(tr.X[tr_cm]).mean(axis=0)
    mu_rpe1 = _to_dense_array(ev.X[ev_cm]).mean(axis=0)

    tr_deltas = _compute_group_deltas(tr[~tr_cm], "perturbation", mu_k562)
    ev_deltas = _compute_group_deltas(ev[~ev_cm], "perturbation", mu_rpe1)

    tr_keys = sorted(tr_deltas.keys())
    ev_keys = sorted(ev_deltas.keys())
    shared_set = set(tr_keys) & set(ev_keys)
    unseen_set = set(ev_keys) - set(tr_keys)

    Y_tr = np.vstack([tr_deltas[p] for p in tr_keys]).astype(np.float32)

    # 1. 提取三教师嵌入并做 L2 归一化拼接
    vocab = sorted(set(tr_keys) | set(ev_keys))
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"

    gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, None)
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)

    def get_fused_emb(plist):
        mats = []
        for d in [gf_dict, scgpt_dict, scf_dict]:
            dim = len(next(iter(d.values())))
            m = np.vstack([d.get(p, np.zeros(dim, dtype=np.float32)) for p in plist]).astype(np.float32)
            mats.append(m)
        full = np.hstack(mats)
        norms = np.linalg.norm(full, axis=1, keepdims=True) + 1e-6
        return full / norms

    E_tr = get_fused_emb(tr_keys)  # [1971, 2432]
    E_ev = get_fused_emb(ev_keys)  # [2016, 2432]

    # 2. 传统 Linear-Ridge 基线
    tr_idx = {p: i for i, p in enumerate(tr_keys)}
    def enc_hot(plist):
        m = np.zeros((len(plist), len(tr_keys)), dtype=np.float32)
        for i, p in enumerate(plist):
            if p in tr_idx:
                m[i, tr_idx[p]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_hot(tr_keys), Y_tr)
    pred_ev_ridge = ridge.predict(enc_hot(ev_keys)).astype(np.float32)

    # 3. 混合特征核岭回归: [One-Hot, alpha * MultiTeacher_Emb]
    # 这样共有基因同时利用自身观测与邻域先验降噪，未见基因自动利用三教师语义插值！
    for w_emb, alpha_r in [(1.0, 20.0), (1.5, 30.0), (2.0, 50.0)]:
        X_tr_hyb = np.hstack([enc_hot(tr_keys), w_emb * E_tr])
        X_ev_hyb = np.hstack([enc_hot(ev_keys), w_emb * E_ev])
        r_hyb = Ridge(alpha=alpha_r, fit_intercept=True)
        r_hyb.fit(X_tr_hyb, Y_tr)
        pred_hyb = r_hyb.predict(X_ev_hyb).astype(np.float32)

        df_hyb = evaluate_perturbation_predictions(ev_deltas, {p: pred_hyb[i] for i, p in enumerate(ev_keys)}, f"TriTeacher-KernelRidge(w={w_emb},a={alpha_r})", (20, 50))
        sh = df_hyb[df_hyb["perturbation"].isin(shared_set)]
        un = df_hyb[df_hyb["perturbation"].isin(unseen_set)]
        print(f"[w={w_emb}, a={alpha_r}] All={df_hyb['pearson_delta_all'].mean():.4f} (Top20={df_hyb['pearson_delta_top20'].mean():.4f}, MSE={df_hyb['mse_all'].mean():.6f}) | Shared={sh['pearson_delta_all'].mean():.4f} | Unseen340={un['pearson_delta_all'].mean():.4f}")

    df_base = evaluate_perturbation_predictions(ev_deltas, {p: pred_ev_ridge[i] for i, p in enumerate(ev_keys)}, "Linear-Ridge", (20, 50))
    sh_b = df_base[df_base["perturbation"].isin(shared_set)]
    un_b = df_base[df_base["perturbation"].isin(unseen_set)]
    print(f"[Linear-Ridge] All={df_base['pearson_delta_all'].mean():.4f} (Top20={df_base['pearson_delta_top20'].mean():.4f}, MSE={df_base['mse_all'].mean():.6f}) | Shared={sh_b['pearson_delta_all'].mean():.4f} | Unseen340={un_b['pearson_delta_all'].mean():.4f}")
