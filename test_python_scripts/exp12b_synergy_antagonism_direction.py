# -*- coding: utf-8 -*-
"""
Exp12b: 多靶点组合的四维机制归因引擎
1. 总效应评估 (Total Effect L2, 1+1>2 协同倍率 Synergy Ratio)
2. 内部主次贡献占比 (Driver vs Passenger Share)
3. 靶点间交互机制定性 (Synergistic 协同 / Antagonistic 拮抗抵消 / Independent 独立叠加)
4. 每个靶点的具体贡献方向 (与总效应的余弦方向 + Top 上调/下调靶基因)
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


def analyze_mechanism(h5ad_path: str, ds_name: str, out_csv: str, device):
    print(f"\n[Exp12b] 分析 {ds_name}: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    pcol = "canon_pert" if "canon_pert" in obs.columns else "perturbation"
    X_all = _to_dense_array(adata.X).astype(np.float32)
    ctrl_mask = (obs[pcol].astype(str) == "control").values
    mu_ctrl = X_all[ctrl_mask].mean(axis=0)
    gene_names = np.array(adata.var_names)

    deltas = {}
    for p, grp in obs[~ctrl_mask].groupby(pcol, observed=True):
        if len(grp) >= 20:
            idx = obs.index.get_indexer(grp.index)
            deltas[str(p)] = X_all[idx].mean(axis=0) - mu_ctrl

    all_keys = sorted(deltas.keys())
    vocab = sorted({t.strip() for k in all_keys for t in k.split("_") if t.strip()})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = 6

    lin_idx = {t: i for i, t in enumerate(vocab)}
    def enc_lin(keys):
        m = np.zeros((len(keys), len(vocab)), dtype=np.float32)
        for ri, ps in enumerate(keys):
            for t in ps.split("_"):
                if t.strip() in lin_idx:
                    m[ri, lin_idx[t.strip()]] = 1.0
        return m

    ridge = Ridge(alpha=10.0, fit_intercept=True).fit(enc_lin(all_keys), np.vstack([deltas[k] for k in all_keys]))
    var_ens = {g: str(adata.var["ensemble_id"].iloc[i]) for i, g in enumerate(adata.var_names)} if "ensemble_id" in adata.var.columns else {}
    gf_dict = extract_geneformer_gene_embeddings(
        r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl",
        r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl",
        vocab, var_ens,
    )
    scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
    scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    coexp_dict = build_gene_coexp_prior(adata.X, list(adata.var_names), prior_dim=64)

    teachers = {"Coexp": coexp_dict, "GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}
    teacher_matrices = []
    for td in teachers.values():
        dim = len(next(iter(td.values())))
        mat = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
        for t, tid in t2id.items():
            if t in td:
                mat[tid] = td[t]
        teacher_matrices.append(mat)

    def make_tids(keys):
        arr = np.zeros((len(keys), max_len), dtype=np.int64)
        for i, p in enumerate(keys):
            for j, t in enumerate(p.split("_")[:max_len]):
                arr[i, j] = t2id.get(t.strip(), 0)
        return arr

    tr_tids = torch.tensor(make_tids(all_keys), device=device)
    tr_dl = torch.tensor(ridge.predict(enc_lin(all_keys)).astype(np.float32), device=device)
    tr_y = torch.tensor(np.vstack([deltas[k] for k in all_keys]).astype(np.float32), device=device)
    tr_xc = torch.tensor(np.tile(mu_ctrl, (len(all_keys), 1)).astype(np.float32), device=device)

    torch.manual_seed(42)
    net = GatedMultiTeacherFusionNet(
        n_genes=adata.n_vars, n_tokens=len(vocab), teacher_matrices=teacher_matrices,
        bottleneck_dim=64, hidden_dim=192, teacher_ctx_dim=1152, dropout=0.20,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    net.train()
    for ep in range(80):
        perm = torch.randperm(len(all_keys), device=device)
        for i in range(0, len(all_keys), 64):
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

    combo_keys = sorted([k for k in all_keys if len(k.split("_")) >= 2])
    full_preds = predict_batch(combo_keys)
    single_preds = {g: predict_batch([g])[0] for g in vocab}

    rows = []
    for idx, combo in enumerate(combo_keys):
        genes = [t.strip() for t in combo.split("_") if t.strip()]
        f_tot = full_preds[idx]
        tot_l2 = float(np.linalg.norm(f_tot))

        # 线性加和基线 (单基因效应之和)
        f_sum = np.sum([single_preds[g] for g in genes], axis=0)
        sum_l2 = float(np.linalg.norm(f_sum)) + 1e-8
        synergy_ratio = tot_l2 / sum_l2
        epistasis_gain = float(np.linalg.norm(f_tot - f_sum))

        # Leave-One-Out 边际贡献向量 v_i = f(S) - f(S \ {g_i})
        loo_keys = ["_".join([g for j, g in enumerate(genes) if j != i]) for i in range(len(genes))]
        loo_preds = predict_batch(loo_keys)
        marg_vecs = f_tot[None, :] - loo_preds  # [K, G]
        marg_l2s = np.linalg.norm(marg_vecs, axis=1)
        shares = marg_l2s / (float(np.sum(marg_l2s)) + 1e-8)

        # 靶点间平均成对余弦相似度 (判断方向协同 vs 拮抗抵消 vs 独立)
        pair_cos = []
        for a in range(len(genes)):
            for b in range(a + 1, len(genes)):
                va, vb = marg_vecs[a], marg_vecs[b]
                c = float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8))
                pair_cos.append(c)
        mean_pair_cos = float(np.mean(pair_cos)) if pair_cos else 0.0

        # 机制定性
        if mean_pair_cos < -0.15 or synergy_ratio < 0.85:
            mech = "Antagonistic (拮抗/部分抵消)"
        elif synergy_ratio > 1.05 or (epistasis_gain / (tot_l2 + 1e-8) > 0.30 and mean_pair_cos >= 0.15):
            mech = "Synergistic (1+1>2 协同增效)"
        elif abs(mean_pair_cos) < 0.15:
            mech = "Independent / Orthogonal (独立正交)"
        else:
            mech = "Additive / Concordant (同向线性叠加)"

        order = np.argsort(-shares)
        drv_i, pas_i = order[0], order[-1]

        def top_sig(vec):
            up_idx = np.argsort(-vec)[:3]
            dn_idx = np.argsort(vec)[:3]
            return "+".join(gene_names[up_idx]), "+".join(gene_names[dn_idx])

        drv_up, drv_dn = top_sig(marg_vecs[drv_i])
        pas_up, pas_dn = top_sig(marg_vecs[pas_i])
        drv_cos_tot = float(np.dot(marg_vecs[drv_i], f_tot) / ((np.linalg.norm(marg_vecs[drv_i]) * tot_l2) + 1e-8))
        pas_cos_tot = float(np.dot(marg_vecs[pas_i], f_tot) / ((np.linalg.norm(marg_vecs[pas_i]) * tot_l2) + 1e-8))

        rows.append({
            "dataset": ds_name,
            "combo": combo,
            "k_genes": len(genes),
            "total_effect_l2": round(tot_l2, 4),
            "sum_singles_l2": round(sum_l2, 4),
            "synergy_ratio_1p1": round(synergy_ratio, 4),
            "epistasis_gain_l2": round(epistasis_gain, 4),
            "inter_target_cos": round(mean_pair_cos, 4),
            "mechanism_type": mech,
            "driver_gene": genes[drv_i],
            "driver_share": round(float(shares[drv_i]), 4),
            "driver_cos_to_total": round(drv_cos_tot, 4),
            "driver_top_up": drv_up,
            "driver_top_down": drv_dn,
            "passenger_gene": genes[pas_i],
            "passenger_share": round(float(shares[pas_i]), 4),
            "passenger_cos_to_total": round(pas_cos_tot, 4),
            "passenger_top_up": pas_up,
            "passenger_top_down": pas_dn,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("机制分布:\n", df["mechanism_type"].value_counts().to_string())
    print("\nTop 5 强协同 (1+1>2) 示例:")
    syn = df[df["mechanism_type"].str.startswith("Synergistic")].sort_values("epistasis_gain_l2", ascending=False).head(5)
    print(syn[["combo", "total_effect_l2", "synergy_ratio_1p1", "inter_target_cos", "driver_gene", "driver_share", "driver_top_up", "passenger_gene", "passenger_share"]].to_string(index=False))
    print("\nTop 5 拮抗/抵消示例:")
    ant = df[df["mechanism_type"].str.startswith("Antagonistic")].sort_values("inter_target_cos").head(5)
    print(ant[["combo", "total_effect_l2", "synergy_ratio_1p1", "inter_target_cos", "driver_gene", "driver_cos_to_total", "passenger_gene", "passenger_cos_to_total"]].to_string(index=False))


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_mechanism(r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad", "Norman2019_K562", r"D:\data\AIVC\test_results\exp12b_norman_mechanism.csv", dev)
    analyze_mechanism(r"D:\data\AIVC\data\Tian2019_iPSC_combo_hvg2000.h5ad", "Tian2019_iPSC", r"D:\data\AIVC\test_results\exp12b_ipsc_mechanism.csv", dev)
