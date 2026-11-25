# -*- coding: utf-8 -*-
"""
exp31_lineage_invariant_gnn_proteome.py
Exp31 公平对比 v1.0 冻结基线 (7-Lineage -> Unseen RPE1, 2,016 扰动):
  1. 解决 Exp30b 退步根因：不把随细胞系剧烈漂移的 x_ctrl 直接灌入 2,665 通路，而是沿 OmniPath 有向激活/抑制边
     传播“谱系不变的 1~2 跳因果扰动足迹”，零 gene_id_emb 过拟合参数。
  2. 接入 Exp28 (aivc_5omics_pathway_gnn.pt) 预测每个细胞系基态的蛋白丰度与磷酸化激活水平，
     作为靶基因门控与状态补充；并检验对 577 个激酶枢纽做 mRNA 掩码、仅靠 Exp28 蛋白/磷酸化预测推断的效果。
"""

import os
import sys
import time
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
from pgnn import build_gene_pathway_bipartite_tensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_8lineage_rpe1_crosscl import extract_lineage_deltas
from exp3_triteacher_distill import load_esm2_gene_embeddings
from exp27b_local_gated_5omics import LocalGated5OmicsPathwayGNN

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRIORS_DIR = r"D:\data\AIVC\data\priors"
EXP28_CKPT = r"D:\data\AIVC\pretrained\aivc_5omics_pathway_gnn.pt"
V1_CSV = r"D:\data\AIVC\versions\v1.0_frozen_baseline\test_results\exp29_cross_lineage_rpe1_results.csv"
OUT_CSV = r"D:\data\AIVC\test_results\exp31_lineage_invariant_gnn_results.csv"


@torch.no_grad()
def predict_lineage_prot_phos_via_exp28(lineage_ctrl_dict: dict, target_genes: list) -> dict:
    """
    利用 Exp28 训练好的 5-Omics 通路图网络，从各细胞系基态转录组预测蛋白丰度与磷酸化状态。
    """
    ck = torch.load(EXP28_CKPT, map_location="cpu", weights_only=False)
    top_rna = [str(g).upper() for g in ck["top_rna"]]
    top_prot = [str(g).upper() for g in ck["top_prot"]]
    top_phos = [str(g).upper() for g in ck["top_phos"]]

    net28 = LocalGated5OmicsPathwayGNN(
        n_rna=len(top_rna),
        n_mut=len(ck["driver_genes"]),
        n_mir=len(ck["top_mir"]),
        rppa_cis=ck["rppa_cis"],
        prot_cis=ck["prot_cis"],
        phos_cis=ck["phos_cis"],
        A_tf=ck["A_tf"],
        A_gp=ck["A_gp"],
        M_rppa=ck["M_rppa"],
        M_prot=ck["M_prot"],
        M_phos=ck["M_phos"],
        hid=160,
    ).to(DEVICE)
    net28.load_state_dict(ck["state_dict"])
    net28.eval()

    sc_g2i = {str(g).upper(): i for i, g in enumerate(target_genes)}
    prot_g2i = {g: i for i, g in enumerate(top_prot)}
    phos_g2i = {g: i for i, g in enumerate(top_phos)}

    out = {}
    for lname, mu_vec in lineage_ctrl_dict.items():
        x_in = np.zeros((1, len(top_rna)), dtype=np.float32)
        mu_z = (mu_vec - mu_vec.mean()) / (mu_vec.std() + 1e-6)
        for j, g in enumerate(top_rna):
            idx = sc_g2i.get(g)
            if idx is not None:
                x_in[0, j] = float(mu_z[idx])
        xt = torch.tensor(x_in, device=DEVICE)
        pred_prot = net28.forward_prot(xt, None, None).squeeze(0).cpu().numpy()
        pred_phos = net28.forward_phos(xt, None, None).squeeze(0).cpu().numpy()

        # 对齐到 target_genes (若未命中则置 0，同时返回是否命中的指示)
        g_prot = np.zeros(len(target_genes), dtype=np.float32)
        g_phos = np.zeros(len(target_genes), dtype=np.float32)
        for idx, g in enumerate(target_genes):
            gu = str(g).upper()
            if gu in prot_g2i:
                g_prot[idx] = float(pred_prot[prot_g2i[gu]])
            if gu in phos_g2i:
                g_phos[idx] = float(pred_phos[phos_g2i[gu]])
        out[lname] = (g_prot, g_phos)

    print(f"[Exp28->Prot/Phos] 已为 {len(out)} 个细胞系预测蛋白({len(top_prot)})与磷酸化({len(top_phos)})基态")
    return out


class LineageInvariantPathwayNet(nn.Module):
    """
    谱系不变的全通路因果足迹 + 蛋白/磷酸化门控网络 (严格保留 v1.0 低秩主干):
      - x_tstat (5维): [raw_mrna, z_mrna, exp28_prot, exp28_phos, log_ridge_norm]
      - x_causal (K维): 扰动靶点沿 OmniPath 有向激活/抑制边传播 1-2 跳后在基因程序空间上的因果足迹
    """

    def __init__(self, n_genes: int, n_teacher: int, k_prog: int, U_prog: np.ndarray, stat_dim: int = 5, hid: int = 128):
        super().__init__()
        self.register_buffer("U_prog", torch.tensor(U_prog, dtype=torch.float32))
        self.ridge_scale = nn.Parameter(torch.ones(1))

        self.target_gate = nn.Sequential(nn.Linear(stat_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.target_gate[-1].weight)
        nn.init.zeros_(self.target_gate[-1].bias)

        self.ctrl_enc = nn.Sequential(nn.Linear(n_genes, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2))
        self.teach_enc = nn.Sequential(nn.Linear(n_teacher, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2))
        self.causal_enc = nn.Sequential(nn.Linear(k_prog, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.25))

        self.prog_head = nn.Sequential(
            nn.Linear(64 + hid + 32 + stat_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hid, k_prog),
        )
        nn.init.zeros_(self.prog_head[-1].weight)
        nn.init.zeros_(self.prog_head[-1].bias)

    def forward(self, x_ctrl, x_teach, x_ridge, x_tstat, x_causal):
        gate = 1.0 + 0.5 * torch.tanh(self.target_gate(x_tstat))
        h = torch.cat([self.ctrl_enc(x_ctrl), self.teach_enc(x_teach), self.causal_enc(x_causal), x_tstat], dim=-1)
        delta_prog = self.prog_head(h) @ self.U_prog
        return gate * (self.ridge_scale * x_ridge) + 0.5 * delta_prog


def main():
    print(f"=== Exp31: 谱系不变 OmniPath 因果足迹 + Exp28 蛋白/磷酸化注入 vs v1.0 ({DEVICE}) ===")
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
        include_biogrid=False,
    )
    A_signed = (bipartite["A_pos"] - bipartite["A_neg"]).astype(np.float32)  # (G, G)
    kinase_mask = bipartite["kinase_like_mask"] > 0.5

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

    lineage_ctrls = {lname: mc for lname, mc, _ in lineage_specs}
    lineage_ctrls["RPE1"] = mu_rpe1
    prot_phos_map = predict_lineage_prot_phos_via_exp28(lineage_ctrls, target_genes)

    tr_samples = []
    for lname, mc, dd in lineage_specs:
        for p, dv in dd.items():
            if "_" in p or "+" in p:
                continue
            tr_samples.append((lname, p, mc, dv))
    ev_keys = sorted(d_rpe1.keys())

    tr_vocab = sorted({s[1] for s in tr_samples})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}

    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, p in enumerate(plist):
            if p in lin_idx:
                m[ri, lin_idx[p]] = 1.0
        return m

    X_tr_lin = enc_lin([s[1] for s in tr_samples])
    Y_tr_lin = np.vstack([s[3] for s in tr_samples]).astype(np.float32)
    ridge = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin, Y_tr_lin)
    pred_ridge_ev = ridge.predict(enc_lin(ev_keys)).astype(np.float32)

    pred_ridge_tr = np.zeros_like(Y_tr_lin)
    lineages_tr = np.array([s[0] for s in tr_samples])
    for ln in np.unique(lineages_tr):
        m_out = (lineages_tr == ln)
        m_in = ~m_out
        if m_in.sum() > 0:
            r_cv = Ridge(alpha=50.0, fit_intercept=True).fit(X_tr_lin[m_in], Y_tr_lin[m_in])
            pred_ridge_tr[m_out] = r_cv.predict(X_tr_lin[m_out]).astype(np.float32)

    svd = TruncatedSVD(n_components=64, random_state=SEED)
    svd.fit(Y_tr_lin)
    U_prog = svd.components_.astype(np.float32)  # (64, G)

    # 预计算全量有向因果足迹：每个基因被扰动后沿 OmniPath (A_pos - A_neg) 传播 1~2 跳并投影到 U_prog
    # 矩阵形式：F_causal = (A_signed + 0.5 * A_signed^2) @ U_prog.T -> (G, 64)
    A1 = A_signed
    A2 = A_signed @ A_signed
    causal_gene_mat = ((A1 + 0.5 * A2) @ U_prog.T).astype(np.float32)  # (G, 64)

    d_coexp = len(next(iter(coexp_dict.values())))
    d_esm = len(next(iter(esm2_dict.values())))

    def build_features(sample_list, ridge_preds, mask_kinase_mrna: bool = False):
        ctrl_l, teach_l, tstat_l, causal_l = [], [], [], []
        for idx, (lname, p, mc, _) in enumerate(sample_list):
            mc_use = mc.copy()
            if mask_kinase_mrna:
                mc_use[kinase_mask] = 0.0
            ctrl_l.append(mc_use)
            c_emb = coexp_dict.get(p, np.zeros(d_coexp, dtype=np.float32))
            e_emb = esm2_dict.get(p, np.zeros(d_esm, dtype=np.float32))
            teach_l.append(np.concatenate([c_emb, e_emb]))

            gi = gene_to_idx.get(p, -1)
            g_prot, g_phos = prot_phos_map[lname]
            if gi >= 0:
                raw_e = float(mc_use[gi])
                z_e = float((mc_use[gi] - mc_use.mean()) / (mc_use.std() + 1e-6))
                pr_v = float(g_prot[gi])
                ph_v = float(g_phos[gi])
                c_vec = causal_gene_mat[gi]
            else:
                raw_e, z_e, pr_v, ph_v = 0.0, 0.0, 0.0, 0.0
                c_vec = np.zeros(64, dtype=np.float32)

            rn = float(np.log1p(np.linalg.norm(ridge_preds[idx])))
            tstat_l.append([raw_e, z_e, pr_v, ph_v, rn])
            causal_l.append(c_vec)
        return (
            np.asarray(ctrl_l, dtype=np.float32),
            np.asarray(teach_l, dtype=np.float32),
            ridge_preds.astype(np.float32),
            np.asarray(tstat_l, dtype=np.float32),
            np.asarray(causal_l, dtype=np.float32),
        )

    def run_variant(vname: str, mask_kinase_mrna: bool):
        t0 = time.time()
        Xc, Xt, Xr, Xs, Xca = build_features(tr_samples, pred_ridge_tr, mask_kinase_mrna=mask_kinase_mrna)
        ev_samples = [("RPE1", k, mu_rpe1, d_rpe1[k]) for k in ev_keys]
        Ec, Et, Er, Es, Eca = build_features(ev_samples, pred_ridge_ev, mask_kinase_mrna=mask_kinase_mrna)

        torch.manual_seed(SEED)
        net = LineageInvariantPathwayNet(n_genes, Xt.shape[1], 64, U_prog, stat_dim=5, hid=128).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        loader = DataLoader(
            TensorDataset(torch.tensor(Xc), torch.tensor(Xt), torch.tensor(Xr), torch.tensor(Xs), torch.tensor(Xca), torch.tensor(Y_tr_lin)),
            batch_size=256,
            shuffle=True,
        )
        net.train()
        for ep in range(45):
            for bc, bt, br, bs, bca, by in loader:
                bc, bt, br, bs, bca, by = bc.to(DEVICE), bt.to(DEVICE), br.to(DEVICE), bs.to(DEVICE), bca.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                pred = net(bc, bt, br, bs, bca)
                loss = F.mse_loss(pred, by) + 0.05 * (1.0 - F.cosine_similarity(pred, by, dim=-1)).mean()
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            pred_ev = net(
                torch.tensor(Ec, device=DEVICE),
                torch.tensor(Et, device=DEVICE),
                torch.tensor(Er, device=DEVICE),
                torch.tensor(Es, device=DEVICE),
                torch.tensor(Eca, device=DEVICE),
            ).cpu().numpy()

        df_ev = evaluate_perturbation_predictions(d_rpe1, {k: pred_ev[i] for i, k in enumerate(ev_keys)}, vname)
        res = {
            "model": vname,
            "mean_pearson": float(df_ev["pearson_delta_all"].mean()),
            "top20_pearson": float(df_ev["pearson_delta_top20"].mean()),
            "dir_acc_top20": float(df_ev["dir_acc_top20"].mean()),
            "mse": float(df_ev["mse_all"].mean()),
        }
        print(f"  [{vname}] ({time.time()-t0:.1f}s) -> {res}")
        return res

    results = []
    if os.path.exists(V1_CSV):
        v1_df = pd.read_csv(V1_CSV)
        for _, row in v1_df.iterrows():
            results.append(dict(row))

    results.append(run_variant("Exp31_OmniPathCausal_ProtPhos", mask_kinase_mrna=False))
    results.append(run_variant("Exp31_OmniPathCausal_MaskedKinase_ProtPhos", mask_kinase_mrna=True))

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Exp31 vs v1.0 对比汇总 (已保存 {OUT_CSV}) ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
