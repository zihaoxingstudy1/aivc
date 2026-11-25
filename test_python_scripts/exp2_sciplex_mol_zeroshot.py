# -*- coding: utf-8 -*-
"""
方向 2 实验：小分子药物多模态结构模型 (ChemBERTa-77M + RDKit Morgan + Z-score Context + 连续剂量)
评测两大核心场景：
1. 跨细胞系药效泛化 (A549 + K562 -> MCF7)
2. 从未见过的全新小分子药物 Zero-shot (150 种药物训练 -> 38 种全新药物测试)
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


class BioMolDrugNet(nn.Module):
    """融合 ChemBERTa 语义嵌入 + RDKit Morgan 分子指纹 + 连续剂量 + 细胞系 Z-score 语境的药物响应模型。"""
    def __init__(self, n_genes: int, mol_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.mol_enc = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dose_enc = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, hidden_dim),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x_ctrl_mean, mol_feat, dose_val, delta_lin):
        xz = (x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(xz)
        z_mol = self.mol_enc(mol_feat) + self.dose_enc(dose_val.unsqueeze(-1))
        res = self.head(torch.cat([z_ctx, z_mol], dim=-1))
        return delta_lin + torch.tanh(self.res_scale) * res


def run_drug_benchmark(adata, train_mask, test_mask, mol_dict, mol_dim, scenario_name, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = adata[train_mask]
    ev = adata[test_mask]

    # Per-Context 提取组 Delta
    def get_ctx_samples(sub):
        samples = []
        for cl in sorted(sub.obs["cell_line"].unique()):
            sub_cl = sub[sub.obs["cell_line"] == cl]
            cm = (sub_cl.obs["pert_dose"].astype(str) == "control").values
            if cm.sum() == 0:
                continue
            mu_c = _to_dense_array(sub_cl.X[cm]).mean(axis=0)
            d_dict = _compute_group_deltas(sub_cl[~cm], "pert_dose", mu_c)
            for p, dt in d_dict.items():
                drug = p.split("__dose_")[0]
                dose = float(np.log10(float(p.split("__dose_")[1]) + 1.0)) if "__dose_" in p else 0.0
                samples.append((cl, p, drug, dose, mu_c, dt))
        return samples

    tr_samples = get_ctx_samples(tr)
    ev_samples = get_ctx_samples(ev)
    print(f"\n=== [{scenario_name}] Train 组数: {len(tr_samples)} | Test 组数: {len(ev_samples)} ===")

    # 1. 基线: Control-Mean / Train-Mean / Linear-Ridge(Drug-ID)
    tr_drugs = sorted({s[2] for s in tr_samples})
    d2idx = {d: i for i, d in enumerate(tr_drugs)}
    def enc_drug_id(samples):
        m = np.zeros((len(samples), len(tr_drugs) + 1), dtype=np.float32)
        for i, s in enumerate(samples):
            if s[2] in d2idx:
                m[i, d2idx[s[2]]] = 1.0
            m[i, -1] = s[3] / 4.0  # 归一化 log10_dose
        return m

    Y_tr = np.vstack([s[5] for s in tr_samples])
    ridge_id = Ridge(alpha=10.0, fit_intercept=True)
    ridge_id.fit(enc_drug_id(tr_samples), Y_tr)
    pred_id = ridge_id.predict(enc_drug_id(ev_samples))

    # 2. 分子结构线性基线: Ridge(ChemBERTa + RDKit Morgan + Dose)
    def enc_mol(samples):
        feats = []
        for s in samples:
            mv = mol_dict.get(s[2], np.zeros(mol_dim, dtype=np.float32))
            feats.append(np.append(mv, s[3] / 4.0))
        return np.vstack(feats).astype(np.float32)

    X_tr_mol = enc_mol(tr_samples)
    X_ev_mol = enc_mol(ev_samples)
    ridge_mol = Ridge(alpha=100.0, fit_intercept=True)
    ridge_mol.fit(X_tr_mol, Y_tr)
    pred_mol_ridge = ridge_mol.predict(X_ev_mol)

    # 构造唯一键值评测 (cl||pert_dose)
    ev_true_dict = {f"{s[0]}||{s[1]}": s[5] for s in ev_samples}
    df_id = evaluate_perturbation_predictions(ev_true_dict, {f"{s[0]}||{s[1]}": pred_id[i] for i, s in enumerate(ev_samples)}, "Linear-Ridge(DrugID)", (20, 50))
    df_mr = evaluate_perturbation_predictions(ev_true_dict, {f"{s[0]}||{s[1]}": pred_mol_ridge[i] for i, s in enumerate(ev_samples)}, "Mol-Ridge(ChemBERTa+Morgan)", (20, 50))

    # 3. 深度多模态模型 BioMolDrugNet (以 Mol-Ridge 或 DrugID 为先验残差)
    tr_xc = torch.tensor(np.vstack([s[4] for s in tr_samples]), device=device, dtype=torch.float32)
    tr_mol = torch.tensor(X_tr_mol[:, :-1], device=device, dtype=torch.float32)
    tr_do = torch.tensor([s[3] for s in tr_samples], device=device, dtype=torch.float32)
    tr_dl = torch.tensor(pred_mol_ridge_tr := ridge_mol.predict(X_tr_mol), device=device, dtype=torch.float32)
    tr_y = torch.tensor(Y_tr, device=device, dtype=torch.float32)

    ev_xc = torch.tensor(np.vstack([s[4] for s in ev_samples]), device=device, dtype=torch.float32)
    ev_mol = torch.tensor(X_ev_mol[:, :-1], device=device, dtype=torch.float32)
    ev_do = torch.tensor([s[3] for s in ev_samples], device=device, dtype=torch.float32)
    ev_dl = torch.tensor(pred_mol_ridge, device=device, dtype=torch.float32)

    net = BioMolDrugNet(adata.n_vars, mol_dim, hidden_dim=192).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(1, 81):
        net.train()
        opt.zero_grad()
        pred = net(tr_xc, tr_mol, tr_do, tr_dl)
        loss = F.mse_loss(pred, tr_y) * 50.0 + (1.0 - F.cosine_similarity(pred, tr_y, dim=-1)).mean() * 0.5
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        epred = net(ev_xc, ev_mol, ev_do, ev_dl).cpu().numpy()
    df_dl = evaluate_perturbation_predictions(ev_true_dict, {f"{s[0]}||{s[1]}": epred[i] for i, s in enumerate(ev_samples)}, "BioMolCell(ChemBERTa+Morgan)", (20, 50))

    all_df = pd.concat([df_id, df_mr, df_dl], ignore_index=True)
    for m, grp in all_df.groupby("model", sort=False):
        print(f"  [{m}] Pearson All={grp['pearson_delta_all'].mean():.4f} | Top20={grp['pearson_delta_top20'].mean():.4f} | MSE={grp['mse_all'].mean():.6f}")
    all_df.to_csv(out_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad")
    npz = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=False)
    drugs = npz["drugs"]
    morgan = npz["morgan"]
    chemberta = npz["chemberta"]
    # L2 归一化拼接分子特征
    m_norm = morgan / (np.linalg.norm(morgan, axis=1, keepdims=True) + 1e-6)
    mol_feats = np.hstack([chemberta, m_norm]).astype(np.float32)
    mol_dict = {str(d): mol_feats[i] for i, d in enumerate(drugs)}
    mol_dim = mol_feats.shape[1]

    # 场景 2A: 跨细胞系 (A549 + K562 -> MCF7)
    run_drug_benchmark(
        adata,
        (adata.obs["ood_split"] == "train").values,
        (adata.obs["ood_split"] == "test").values,
        mol_dict, mol_dim,
        "sci-Plex3 Cross-CellLine (A549+K562 -> MCF7)",
        r"D:\data\AIVC\test_results\exp2a_sciplex_crosscl.csv",
    )

    # 场景 2B: 全新未见小分子药物 Zero-shot (按药物互斥 80/20 切分)
    rng = np.random.default_rng(42)
    all_d = sorted([d for d in adata.obs["perturbation"].astype(str).unique() if d != "control"])
    perm_d = rng.permutation(all_d)
    test_drugs = set(perm_d[:38])
    is_ctrl = (adata.obs["perturbation"].astype(str) == "control").values
    in_test_drug = adata.obs["perturbation"].astype(str).isin(test_drugs).values
    # Control 按细胞系 80/20 分
    ctrl_test_mask = np.zeros(adata.n_obs, dtype=bool)
    ctrl_idx = np.where(is_ctrl)[0]
    ctrl_test_mask[rng.choice(ctrl_idx, size=int(0.2 * len(ctrl_idx)), replace=False)] = True

    tr_mask_2b = (~in_test_drug) & (~ctrl_test_mask)
    te_mask_2b = in_test_drug | ctrl_test_mask
    run_drug_benchmark(
        adata,
        tr_mask_2b,
        te_mask_2b,
        mol_dict, mol_dim,
        "sci-Plex3 Unseen-Drug Zero-Shot (150 Drugs -> 38 New Drugs)",
        r"D:\data\AIVC\test_results\exp2b_sciplex_unseendrug.csv",
    )
