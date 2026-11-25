# -*- coding: utf-8 -*-
"""
方向 7 升级 (Exp12): 跨 4 大数据集 (Norman2019 K562 + Tian2019 iPSC + Dixit2016 K562 + Wessels2023 THP1)
合并双基因 (2-gene) 与多基因 (3+ gene) 组合，统一识别 Driver (驱动靶点) vs Passenger (乘客/跟随靶点) 规律！
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings
from bfc import build_gene_coexp_prior
from omni_fusion import GatedMultiTeacherFusionNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_triteacher_distill import load_scgpt_gene_embeddings, load_esm2_gene_embeddings, load_scfoundation_gene_embeddings


def clean_gene_token(tok: str, mode: str) -> str:
    t = str(tok).strip()
    if not t or t.lower() in ("control", "nan", "none"):
        return ""
    if mode == "dixit":
        if "INTERGENIC" in t.upper():
            return ""
        t = re.sub(r"^p-sg", "", t)
        t = re.sub(r"-\d+$", "", t)
    return t


def prepare_dataset_combos(h5ad_path: str, name: str, mode: str, min_cells: int = 20):
    print(f"[Exp12] 加载 {name}: {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    # 标准化 (若尚未 log1p)
    x_max = float(adata.X[:500].max()) if adata.n_obs > 500 else float(adata.X.max())
    if x_max > 25.0:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if adata.n_vars > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

    obs = adata.obs.copy()
    def to_canon(p):
        if str(p).lower() == "control":
            return "control"
        toks = sorted({clean_gene_token(x, mode) for x in str(p).split("_")})
        toks = [t for t in toks if t]
        return "_".join(toks) if toks else "drop"

    obs["canon_pert"] = obs["perturbation"].astype(str).map(to_canon)
    obs["n_canon"] = obs["canon_pert"].map(lambda x: 0 if x == "control" else (-1 if x == "drop" else len(x.split("_"))))
    vc = obs["canon_pert"].value_counts()
    valid = set(vc[vc >= min_cells].index)
    keep = (obs["canon_pert"] == "control") | (obs["canon_pert"].isin(valid) & obs["n_canon"].between(1, 6))
    adata = adata[keep].copy()
    obs = adata.obs.copy()
    obs["canon_pert"] = obs["perturbation"].astype(str).map(to_canon)
    obs["n_canon"] = obs["canon_pert"].map(lambda x: 0 if x == "control" else len(x.split("_")))

    X_all = _to_dense_array(adata.X).astype(np.float32)
    ctrl_mask = (obs["canon_pert"] == "control").values
    mu_ctrl = X_all[ctrl_mask].mean(axis=0)

    deltas, n_map = {}, {}
    for cp, grp in obs[~ctrl_mask].groupby("canon_pert", observed=True):
        idx = obs.index.get_indexer(grp.index)
        deltas[str(cp)] = X_all[idx].mean(axis=0) - mu_ctrl
        n_map[str(cp)] = int(grp["n_canon"].iloc[0])

    singles = [k for k, n in n_map.items() if n == 1]
    combos = [k for k, n in n_map.items() if n >= 2]
    print(f"  -> {name}: 单基因={len(singles)}, 双基因+多基因组合(2+3+...)={len(combos)}")
    return adata, obs, mu_ctrl, deltas, n_map


def analyze_one_dataset(name: str, h5ad_path: str, mode: str, device):
    adata, obs, mu_ctrl, deltas, n_map = prepare_dataset_combos(h5ad_path, name, mode, min_cells=20)
    all_keys = sorted(deltas.keys())
    vocab = sorted({t for k in all_keys for t in k.split("_")})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = 6

    lin_idx = {t: i for i, t in enumerate(vocab)}
    def enc_lin(keys):
        m = np.zeros((len(keys), len(vocab)), dtype=np.float32)
        for ri, ps in enumerate(keys):
            for t in ps.split("_"):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
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
                arr[i, j] = t2id.get(t, 0)
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

    combo_keys = sorted([k for k, n in n_map.items() if n >= 2])
    full_preds = predict_batch(combo_keys)
    gene_contribs = {g: [] for g in vocab}
    decomp_rows = []

    for idx, combo in enumerate(combo_keys):
        genes = combo.split("_")
        loo_keys = ["_".join([g for j, g in enumerate(genes) if j != i]) for i in range(len(genes))]
        loo_preds = predict_batch(loo_keys)
        marg = np.linalg.norm(full_preds[idx][None, :] - loo_preds, axis=1)
        total_m = float(np.sum(marg)) + 1e-8
        shares = marg / total_m
        order = np.argsort(-shares)
        for g, sh, mv in zip(genes, shares, marg):
            gene_contribs[g].append((float(sh), float(mv), len(genes)))
        decomp_rows.append({
            "dataset": name,
            "combo": combo,
            "k_genes": len(genes),
            "driver_gene": genes[order[0]],
            "driver_share": float(shares[order[0]]),
            "passenger_gene": genes[order[-1]],
            "passenger_share": float(shares[order[-1]]),
            "dominance_gap": float(shares[order[0]] - shares[order[-1]]),
        })

    rank_rows = []
    for g, vals in gene_contribs.items():
        if vals:
            shs = [v[0] for v in vals]
            mvs = [v[1] for v in vals]
            ks = [v[2] for v in vals]
            # 归一化主导度 (乘以平均组合大小 K，1.0 表示平均水平，>1.25 为 Driver，<0.80 为 Passenger)
            norm_dom = float(np.mean([s * k for s, k in zip(shs, ks)]))
            role = "Master Driver" if norm_dom >= 1.25 else ("Co-Driver" if norm_dom >= 0.85 else "Passenger / Follower")
            rank_rows.append({
                "dataset": name,
                "target_gene": g,
                "n_combos": len(vals),
                "mean_raw_share": float(np.mean(shs)),
                "normalized_driver_score": norm_dom,
                "mean_marginal_l2": float(np.mean(mvs)),
                "role": role,
            })

    return rank_rows, decomp_rows


def run_exp12():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [
        ("Norman2019_K562", r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad", "std"),
        ("Tian2019_iPSC", r"D:\data\AIVC\data\Tian2019_iPSC_combo_hvg2000.h5ad", "std"),
        ("Dixit2016_K562_MOI", r"D:\data\AIVC\data\DixitRegev2016_K562_TFs_High_MOI.h5ad", "dixit"),
        ("Wessels2023_THP1", r"D:\data\AIVC\data\WesselsSatija2023.h5ad", "std"),
    ]
    all_rank, all_decomp = [], []
    for name, path, mode in specs:
        if os.path.exists(path):
            r_rows, d_rows = analyze_one_dataset(name, path, mode, device)
            all_rank.extend(r_rows)
            all_decomp.extend(d_rows)

    rank_df = pd.DataFrame(all_rank).sort_values(["dataset", "normalized_driver_score"], ascending=[True, False])
    decomp_df = pd.DataFrame(all_decomp)
    rank_df.to_csv(r"D:\data\AIVC\test_results\exp12_unified_driver_passenger_ranking.csv", index=False, encoding="utf-8")
    decomp_df.to_csv(r"D:\data\AIVC\test_results\exp12_unified_combo_decompositions.csv", index=False, encoding="utf-8")

    print("\n=== Exp12 跨 4 大数据集 (2-gene + 3+ gene 合并) Driver vs Passenger 汇总 ===")
    print(f"总计拆解组合数: {len(decomp_df)} (2-gene: {(decomp_df['k_genes']==2).sum()}, 3+ gene: {(decomp_df['k_genes']>=3).sum()})")
    print("角色分布:\n", rank_df.groupby(["dataset", "role"]).size().unstack(fill_value=0))
    for ds, g in rank_df.groupby("dataset"):
        top_drv = g.head(3)["target_gene"].tolist()
        bot_pas = g.tail(3)["target_gene"].tolist()
        print(f"  [{ds}] Top Drivers={top_drv} | Top Passengers={bot_pas}")

    # 绘图
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    sns.boxplot(data=decomp_df, x="dataset", y="driver_share", color="#e74c3c", width=0.35, ax=axes[0], boxprops={"alpha": 0.75})
    sns.boxplot(data=decomp_df, x="dataset", y="passenger_share", color="#3498db", width=0.35, ax=axes[0], boxprops={"alpha": 0.75})
    axes[0].set_title("Driver vs. Passenger Contribution Share Across 4 Datasets\n(Red=Dominant Driver, Blue=Follower Passenger)", fontweight="bold")
    axes[0].set_ylabel("Leave-One-Out Marginal Contribution Share")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    top_all = rank_df[rank_df["n_combos"] >= 5].sort_values("normalized_driver_score", ascending=False)
    show_df = pd.concat([top_all.head(10), top_all.tail(8)])
    show_df["label"] = show_df["target_gene"] + " (" + show_df["dataset"].str.split("_").str[0] + ")"
    colors = ["#e74c3c" if r == "Master Driver" else ("#f39c12" if r == "Co-Driver" else "#3498db") for r in show_df["role"]]
    axes[1].barh(show_df["label"][::-1], show_df["normalized_driver_score"][::-1], color=colors[::-1])
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Equal-Share Baseline (1.0)")
    axes[1].set_title("Top Master Drivers vs. Follower Passengers\n(Normalized Driver Score = Share × K)", fontweight="bold")
    axes[1].set_xlabel("Normalized Driver Dominance Score")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(r"D:\data\AIVC\test_results\exp12_driver_vs_passenger_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("[Exp12] 全部结果与图表已保存至 exp12_unified_driver_passenger_ranking.csv / exp12_driver_vs_passenger_summary.png")


if __name__ == "__main__":
    run_exp12()
