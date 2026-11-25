# -*- coding: utf-8 -*-
"""
exp30_pathway_gnn_base_vs_v1.py
Step 2 对比验证：全量通路“基因-通路二部图”GNN 底座 vs v1.0 冻结基线 (7-Lineage -> Unseen RPE1)
同时检验当所有激酶/信号枢纽节点被强制掩码为 [MASK] 未知态时，模型能否依然保持泛化能力并自行推断状态。
"""

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
from pgnn import build_gene_pathway_bipartite_tensors, PathwayGeneBipartiteNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRIORS_DIR = r"D:\data\AIVC\data\priors"
V1_CSV = r"D:\data\AIVC\versions\v1.0_frozen_baseline\test_results\exp29_cross_lineage_rpe1_results.csv"
OUT_CSV = r"D:\data\AIVC\test_results\exp30_pathway_gnn_vs_v1_results.csv"


def train_and_eval_variant(
    name: str,
    bipartite: dict,
    U_prog: np.ndarray,
    tr_tensors: tuple,
    ev_tensors: tuple,
    d_rpe1: dict,
    ev_keys: list,
    use_latent_mask: bool = False,
    mask_prob_train: float = 0.0,
    eval_force_mask_kinases: bool = False,
    epochs: int = 18,
    batch_size: int = 128,
) -> dict:
    X_ctrl_tr, X_teach_tr, X_ridge_tr, X_tstat_tr, X_pert_tr, Y_tr = tr_tensors
    X_ctrl_ev, X_teach_ev, X_ridge_ev, X_tstat_ev, X_pert_ev = ev_tensors

    torch.manual_seed(SEED)
    model = PathwayGeneBipartiteNet(
        n_genes=X_ctrl_tr.shape[1],
        n_teacher=X_teach_tr.shape[1],
        U_prog=U_prog,
        bipartite_tensors=bipartite,
        d_node=24,
        hid=128,
        use_latent_mask=use_latent_mask,
        mask_prob_train=mask_prob_train,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_ctrl_tr),
            torch.tensor(X_teach_tr),
            torch.tensor(X_ridge_tr),
            torch.tensor(X_tstat_tr),
            torch.tensor(X_pert_tr),
            torch.tensor(Y_tr),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    for ep in range(epochs):
        model.train()
        for bx_c, bx_t, bx_r, bx_s, bx_p, by in loader:
            bx_c = bx_c.to(DEVICE)
            bx_t = bx_t.to(DEVICE)
            bx_r = bx_r.to(DEVICE)
            bx_s = bx_s.to(DEVICE)
            bx_p = bx_p.to(DEVICE)
            by = by.to(DEVICE)

            opt.zero_grad()
            pred, _ = model(bx_c, bx_t, bx_r, bx_s, bx_p, force_mask_kinases=False)
            loss_mse = F.mse_loss(pred, by)
            cos = F.cosine_similarity(pred, by, dim=-1).mean()
            loss = loss_mse + 0.02 * (1.0 - cos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_ctrl_ev), 256):
            p, _ = model(
                torch.tensor(X_ctrl_ev[i:i+256]).to(DEVICE),
                torch.tensor(X_teach_ev[i:i+256]).to(DEVICE),
                torch.tensor(X_ridge_ev[i:i+256]).to(DEVICE),
                torch.tensor(X_tstat_ev[i:i+256]).to(DEVICE),
                torch.tensor(X_pert_ev[i:i+256]).to(DEVICE),
                force_mask_kinases=eval_force_mask_kinases,
            )
            preds.append(p.cpu().numpy())
    pred_mat = np.vstack(preds)
    ev_pred = {k: pred_mat[i] for i, k in enumerate(ev_keys)}
    ev_true = {k: d_rpe1[k] for k in ev_keys}
    df_eval = evaluate_perturbation_predictions(ev_true, ev_pred, name)
    res = {
        "model": name,
        "mean_pearson": float(df_eval["pearson_delta_all"].mean()),
        "top20_pearson": float(df_eval["pearson_delta_top20"].mean()),
        "dir_acc_top20": float(df_eval["dir_acc_top20"].mean()),
        "mse": float(df_eval["mse_all"].mean()),
    }
    print(f"  [{name}] -> {res}")
    return res


def main():
    print(f"=== Exp30: 全通路二部图 GNN 底座 vs v1.0 冻结基线 ({DEVICE}) ===")
    ref_h5ad = r"D:\data\AIVC\data\Replogle_K562_RPE1_hvg2000_split.h5ad"
    ref = sc.read_h5ad(ref_h5ad)
    target_genes = list(ref.var_names)
    gene_to_idx = {g: i for i, g in enumerate(target_genes)}
    n_genes = len(target_genes)

    tr_ref = ref[ref.obs["ood_split"] == "train"]
    ev_ref = ref[ref.obs["ood_split"] == "test"]
    mu_k562 = _to_dense_array(tr_ref[tr_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    mu_rpe1 = _to_dense_array(ev_ref[ev_ref.obs["perturbation"] == "control"].X).mean(axis=0).astype(np.float32)
    d_k562 = _compute_group_deltas(tr_ref[tr_ref.obs["perturbation"] != "control"], "perturbation", mu_k562)
    d_rpe1 = _compute_group_deltas(ev_ref[ev_ref.obs["perturbation"] != "control"], "perturbation", mu_rpe1)

    coexp_dict = build_gene_coexp_prior(tr_ref.X, target_genes, prior_dim=64)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    del ref, tr_ref, ev_ref

    bipartite = build_gene_pathway_bipartite_tensors(
        target_genes=target_genes,
        priors_dir=PRIORS_DIR,
        min_genes_per_pathway=3,
        include_biogrid=True,
    )

    lineage_specs = [("K562", mu_k562, d_k562)]
    extra_files = [
        ("HepG2", r"D:\data\AIVC\data\NadigOConner2024_hepg2.h5ad", False),
        ("Jurkat", r"D:\data\AIVC\data\NadigOConner2024_jurkat.h5ad", False),
        ("iPSC", r"D:\data\AIVC\data\TianKampmann2019_iPSC.h5ad", True),
        ("Norman", r"D:\data\AIVC\data\NormanWeissman2019_filtered.h5ad", True),
        ("PrimaryT", r"D:\data\AIVC\data\ShifrutMarson2018.h5ad", True),
        ("THP1", r"D:\data\AIVC\data\PapalexiSatija2021_eccite_RNA.h5ad", True),
    ]
    for lname, fpath, sp_only in extra_files:
        if os.path.exists(fpath):
            mc, dd = extract_lineage_deltas(fpath, target_genes, lname, min_cells=10, single_pert_only=sp_only)
            lineage_specs.append((lname, mc, dd))

    tr_samples = []
    for lname, mc, dd in lineage_specs:
        for p, dv in dd.items():
            if "_" in p or "+" in p:
                continue
            tr_samples.append((lname, p, mc, dv))
    ev_keys = sorted(d_rpe1.keys())

    # Cross-lineage Ridge prior + SVD K=64 gene programs
    tr_vocab = sorted({s[1] for s in tr_samples})
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
    pred_ridge_tr = ridge.predict(X_tr_lin).astype(np.float32)
    pred_ridge_ev = ridge.predict(enc_lin(ev_keys)).astype(np.float32)

    svd = TruncatedSVD(n_components=64, random_state=SEED)
    svd.fit(Y_tr_lin)
    U_prog = svd.components_.astype(np.float32)

    d_coexp = len(next(iter(coexp_dict.values())))
    d_esm = len(next(iter(esm2_dict.values())))

    def make_inputs(sample_list, ridge_preds):
        ctrl_l, teach_l, tstat_l, pert_l = [], [], [], []
        for idx, (lname, p, mc, _) in enumerate(sample_list):
            ctrl_l.append(mc)
            c_emb = coexp_dict.get(p, np.zeros(d_coexp, dtype=np.float32))
            e_emb = esm2_dict.get(p, np.zeros(d_esm, dtype=np.float32))
            teach_l.append(np.concatenate([c_emb, e_emb]))
            gi = gene_to_idx.get(p, -1)
            pert_vec = np.zeros(n_genes, dtype=np.float32)
            if gi >= 0:
                pert_vec[gi] = 1.0
                raw_e = float(mc[gi])
                z_e = float((mc[gi] - mc.mean()) / (mc.std() + 1e-6))
            else:
                raw_e, z_e = 0.0, 0.0
            rn = float(np.log1p(np.linalg.norm(ridge_preds[idx])))
            tstat_l.append([raw_e, z_e, rn])
            pert_l.append(pert_vec)
        return (
            np.asarray(ctrl_l, dtype=np.float32),
            np.asarray(teach_l, dtype=np.float32),
            ridge_preds.astype(np.float32),
            np.asarray(tstat_l, dtype=np.float32),
            np.asarray(pert_l, dtype=np.float32),
        )

    X_ctrl_tr, X_teach_tr, X_ridge_tr, X_tstat_tr, X_pert_tr = make_inputs(tr_samples, pred_ridge_tr)
    ev_samples = [("RPE1", k, mu_rpe1, d_rpe1[k]) for k in ev_keys]
    X_ctrl_ev, X_teach_ev, X_ridge_ev, X_tstat_ev, X_pert_ev = make_inputs(ev_samples, pred_ridge_ev)

    tr_tensors = (X_ctrl_tr, X_teach_tr, X_ridge_tr, X_tstat_tr, X_pert_tr, Y_tr_lin.astype(np.float32))
    ev_tensors = (X_ctrl_ev, X_teach_ev, X_ridge_ev, X_tstat_ev, X_pert_ev)

    results = []
    if os.path.exists(V1_CSV):
        v1_df = pd.read_csv(V1_CSV)
        for _, row in v1_df.iterrows():
            results.append(dict(row))

    results.append(
        train_and_eval_variant(
            "Exp30_PathwayGNN_Base",
            bipartite, U_prog, tr_tensors, ev_tensors, d_rpe1, ev_keys,
            use_latent_mask=False, mask_prob_train=0.0, eval_force_mask_kinases=False,
        )
    )
    results.append(
        train_and_eval_variant(
            "Exp30_PathwayGNN_MaskedKinase_EvalFull",
            bipartite, U_prog, tr_tensors, ev_tensors, d_rpe1, ev_keys,
            use_latent_mask=True, mask_prob_train=0.5, eval_force_mask_kinases=False,
        )
    )
    results.append(
        train_and_eval_variant(
            "Exp30_PathwayGNN_MaskedKinase_EvalMasked",
            bipartite, U_prog, tr_tensors, ev_tensors, d_rpe1, ev_keys,
            use_latent_mask=True, mask_prob_train=0.5, eval_force_mask_kinases=True,
        )
    )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== 对比汇总 (已保存 {OUT_CSV}) ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
