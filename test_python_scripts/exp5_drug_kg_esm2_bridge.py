# -*- coding: utf-8 -*-
"""
方向 2 终极升级 (Exp5): 药物-靶点蛋白知识图谱 (Drug-Target ESM-2 + Pathway) 跨模态桥接
在 sci-Plex3 Unseen Drug Zero-Shot 上对比：
1. Linear-Ridge (DrugID)
2. BioMolCell (纯分子结构 ChemBERTa + Morgan)
3. BioMolKG-Blind (仅给 SMILES，通过训练好的 Mol2Target-ESM2 跨模态桥接器推断虚拟靶点与通路嵌入)
4. BioMolKG-TargetKnown (已知靶点先验的先导化合物优化场景)
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array


@torch.no_grad()
def build_drug_target_esm2_embeddings(smiles_csv: str, esm2_dir: str, cache_npz: str) -> tuple:
    if os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=False)
        return {str(k): d["target_embs"][i] for i, k in enumerate(d["targets"])}, {str(k): d["pathway_onehot"][i] for i, k in enumerate(d["drugs"])}

    df = pd.read_csv(smiles_csv)
    targets = sorted(df["target"].fillna("Others").astype(str).unique())
    pathways = sorted(df["pathway_level_1"].fillna("Other").astype(str).unique())
    pw_idx = {p: i for i, p in enumerate(pathways)}

    # 从 UniProt 查询每个 target 关键词对应的人类蛋白序列
    target_to_seq = {}
    for tg in targets:
        clean_tg = tg.split("(")[0].split(",")[0].strip()
        if clean_tg in ("Others", "Other", "DNA/RNA Synthesis", "DNA alkylator"):
            continue
        q = f"({clean_tg}) AND organism_id:9606 AND reviewed:true"
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({"query": q, "fields": "sequence", "format": "json", "size": 1})
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            res = json.loads(resp.read().decode("utf-8")).get("results", [])
            if res:
                seq = res[0].get("sequence", {}).get("value", "")
                if seq:
                    target_to_seq[tg] = seq[:1022]
        except Exception:
            continue

    print(f"[DrugKG] UniProt 成功匹配药物靶点蛋白序列: {len(target_to_seq)}/{len(targets)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(esm2_dir)
    esm = AutoModel.from_pretrained(esm2_dir, torch_dtype=torch.float16).to(device)
    esm.eval()

    tg_keys = sorted(target_to_seq.keys())
    seqs = [target_to_seq[k] for k in tg_keys]
    embs_list = []
    for i in range(0, len(seqs), 16):
        enc = tok(seqs[i : i + 16], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
        out = esm(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state.float() * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        emb = F.normalize(emb, dim=-1)
        embs_list.append(emb.cpu().numpy().astype(np.float32))

    del esm
    torch.cuda.empty_cache()

    emb_mat = np.vstack(embs_list) if embs_list else np.zeros((0, 640), dtype=np.float32)
    tg_emb_dict = {k: emb_mat[i] for i, k in enumerate(tg_keys)}

    # 保存所有 target 映射 (未命中的置零)
    all_tg_mat = np.vstack([tg_emb_dict.get(t, np.zeros(640, dtype=np.float32)) for t in targets]).astype(np.float32)
    drugs = df["perturbation"].astype(str).tolist()
    pw_mat = np.zeros((len(drugs), len(pathways)), dtype=np.float32)
    for i, pw in enumerate(df["pathway_level_1"].fillna("Other").astype(str)):
        pw_mat[i, pw_idx[pw]] = 1.0

    np.savez(cache_npz, targets=np.array(targets), target_embs=all_tg_mat, drugs=np.array(drugs), pathway_onehot=pw_mat)
    return {t: all_tg_mat[i] for i, t in enumerate(targets)}, {d: pw_mat[i] for i, d in enumerate(drugs)}


class Mol2TargetBridge(nn.Module):
    """跨模态药物-靶点知识图谱桥接器：从分子结构 (ChemBERTa+Morgan) 预测 ESM2 靶点蛋白嵌入与通路分布。"""
    def __init__(self, mol_dim: int, esm_dim: int, pw_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(mol_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.esm_head = nn.Linear(256, esm_dim)
        self.pw_head = nn.Linear(256, pw_dim)

    def forward(self, mol_feat):
        h = self.enc(mol_feat)
        return F.normalize(self.esm_head(h), dim=-1), self.pw_head(h)


class BioMolKGNet(nn.Module):
    """融合分子结构 + 靶点蛋白 ESM-2 + 通路知识图谱 + 剂量 + Z-score 细胞基态的药物响应网络。"""
    def __init__(self, n_genes: int, mol_dim: int, kg_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.mol_enc = nn.Sequential(nn.Linear(mol_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.25))
        self.kg_enc = nn.Sequential(nn.Linear(kg_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.25))
        self.dose_enc = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, hidden_dim))
        self.ctx_enc = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x_ctrl, mol_f, kg_f, dose_v, delta_lin):
        xz = (x_ctrl - x_ctrl.mean(dim=-1, keepdim=True)) / (x_ctrl.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(xz)
        z_mol = self.mol_enc(mol_f) + self.dose_enc(dose_v.unsqueeze(-1))
        z_kg = self.kg_enc(kg_f)
        res = self.head(torch.cat([z_ctx, z_mol, z_kg], dim=-1))
        return delta_lin + torch.tanh(self.res_scale) * res


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles_csv = r"D:\data\AIVC\data\sciplex3_drug_smiles.csv"
    df_meta = pd.read_csv(smiles_csv)
    drug_to_target = {str(r["perturbation"]): str(r["target"]) for _, r in df_meta.iterrows()}

    tg_emb_dict, pw_dict = build_drug_target_esm2_embeddings(
        smiles_csv, r"D:\data\AIVC\pretrained\esm2_150m", r"D:\data\AIVC\data\sciplex3_target_esm2_kg.npz"
    )

    npz = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=False)
    m_norm = npz["morgan"] / (np.linalg.norm(npz["morgan"], axis=1, keepdims=True) + 1e-6)
    mol_feats = np.hstack([npz["chemberta"], m_norm]).astype(np.float32)
    mol_dict = {str(d): mol_feats[i] for i, d in enumerate(npz["drugs"])}
    mol_dim = mol_feats.shape[1]

    # 构造每个药物的真实 KG 向量: [ESM2_Target (640d), Pathway_OneHot (16d)]
    kg_dict = {}
    for d in mol_dict:
        tg = drug_to_target.get(d, "Others")
        e_vec = tg_emb_dict.get(tg, np.zeros(640, dtype=np.float32))
        p_vec = pw_dict.get(d, np.zeros(16, dtype=np.float32))
        kg_dict[d] = np.concatenate([e_vec, p_vec], axis=0).astype(np.float32)
    kg_dim = len(next(iter(kg_dict.values())))

    # 加载 sci-Plex3 数据并按药物互斥 80/20 切分 (与 exp2b 完全同口径！)
    adata = sc.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad")
    rng = np.random.default_rng(42)
    all_d = sorted([d for d in adata.obs["perturbation"].astype(str).unique() if d != "control"])
    perm_d = rng.permutation(all_d)
    train_drugs = set(perm_d[38:])
    test_drugs = set(perm_d[:38])

    # 1. 在 150 个训练药物上训练 Mol2TargetBridge (完全不看 38 个测试药物！)
    tr_d_list = sorted(train_drugs)
    te_d_list = sorted(test_drugs)
    X_m_tr = torch.tensor(np.vstack([mol_dict[d] for d in tr_d_list]), device=device)
    Y_esm_tr = torch.tensor(np.vstack([kg_dict[d][:640] for d in tr_d_list]), device=device)
    Y_pw_tr = torch.tensor(np.vstack([kg_dict[d][640:] for d in tr_d_list]), device=device)

    bridge = Mol2TargetBridge(mol_dim, 640, 16).to(device)
    b_opt = torch.optim.AdamW(bridge.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(150):
        bridge.train()
        b_opt.zero_grad()
        p_esm, p_pw = bridge(X_m_tr)
        # 仅对有有效靶点序列的样本算 ESM2 对齐
        valid_e = (Y_esm_tr.norm(dim=-1) > 0.1)
        loss_e = (1.0 - F.cosine_similarity(p_esm[valid_e], Y_esm_tr[valid_e], dim=-1)).mean() if valid_e.sum() > 0 else 0.0
        loss_pw = F.cross_entropy(p_pw, Y_pw_tr.argmax(dim=-1))
        (loss_e + 0.5 * loss_pw).backward()
        b_opt.step()

    bridge.eval()
    with torch.no_grad():
        X_m_te = torch.tensor(np.vstack([mol_dict[d] for d in te_d_list]), device=device)
        pred_esm_te, pred_pw_te = bridge(X_m_te)
        pred_pw_prob = F.softmax(pred_pw_te, dim=-1).cpu().numpy()
        pred_esm_np = pred_esm_te.cpu().numpy()
        pw_acc = float(np.mean(pred_pw_prob.argmax(axis=-1) == np.vstack([kg_dict[d][640:] for d in te_d_list]).argmax(axis=-1)))
        print(f"[DrugKG-Bridge] 在 38 个全新未见药物上的通路盲测准确率: {pw_acc*100:.1f}%")

    blind_kg_dict = {d: kg_dict[d] for d in tr_d_list}
    for i, d in enumerate(te_d_list):
        blind_kg_dict[d] = np.concatenate([pred_esm_np[i], pred_pw_prob[i]], axis=0).astype(np.float32)

    # 2. 切分数据并评估
    is_ctrl = (adata.obs["perturbation"].astype(str) == "control").values
    in_test = adata.obs["perturbation"].astype(str).isin(test_drugs).values
    ctrl_te_mask = np.zeros(adata.n_obs, dtype=bool)
    ctrl_te_mask[rng.choice(np.where(is_ctrl)[0], size=int(0.2 * is_ctrl.sum()), replace=False)] = True

    tr_sub = adata[(~in_test) & (~ctrl_te_mask)]
    te_sub = adata[in_test | ctrl_te_mask]

    def get_samples(sub):
        samples = []
        for cl in sorted(sub.obs["cell_line"].unique()):
            sub_cl = sub[sub.obs["cell_line"] == cl]
            cm = (sub_cl.obs["pert_dose"].astype(str) == "control").values
            mu_c = _to_dense_array(sub_cl.X[cm]).mean(axis=0)
            dd = _compute_group_deltas(sub_cl[~cm], "pert_dose", mu_c)
            for p, dt in dd.items():
                drug = p.split("__dose_")[0]
                dose = float(np.log10(float(p.split("__dose_")[1]) + 1.0)) if "__dose_" in p else 0.0
                samples.append((cl, p, drug, dose, mu_c, dt))
        return samples

    tr_s = get_samples(tr_sub)
    te_s = get_samples(te_sub)
    Y_tr = np.vstack([s[5] for s in tr_s])
    ev_true = {f"{s[0]}||{s[1]}": s[5] for s in te_s}

    def run_kg_model(use_kg_dict, name):
        X_tr_f = np.vstack([np.concatenate([mol_dict[s[2]], use_kg_dict[s[2]], [s[3] / 4.0]]) for s in tr_s]).astype(np.float32)
        X_te_f = np.vstack([np.concatenate([mol_dict[s[2]], use_kg_dict[s[2]], [s[3] / 4.0]]) for s in te_s]).astype(np.float32)
        ridge = Ridge(alpha=100.0, fit_intercept=True)
        ridge.fit(X_tr_f, Y_tr)
        dl_tr = torch.tensor(ridge.predict(X_tr_f).astype(np.float32), device=device)
        dl_te = torch.tensor(ridge.predict(X_te_f).astype(np.float32), device=device)

        xc_tr = torch.tensor(np.vstack([s[4] for s in tr_s]), device=device, dtype=torch.float32)
        mol_tr = torch.tensor(np.vstack([mol_dict[s[2]] for s in tr_s]), device=device, dtype=torch.float32)
        kg_tr = torch.tensor(np.vstack([use_kg_dict[s[2]] for s in tr_s]), device=device, dtype=torch.float32)
        do_tr = torch.tensor([s[3] for s in tr_s], device=device, dtype=torch.float32)
        yt = torch.tensor(Y_tr, device=device, dtype=torch.float32)

        xc_te = torch.tensor(np.vstack([s[4] for s in te_s]), device=device, dtype=torch.float32)
        mol_te = torch.tensor(np.vstack([mol_dict[s[2]] for s in te_s]), device=device, dtype=torch.float32)
        kg_te = torch.tensor(np.vstack([use_kg_dict[s[2]] for s in te_s]), device=device, dtype=torch.float32)
        do_te = torch.tensor([s[3] for s in te_s], device=device, dtype=torch.float32)

        torch.manual_seed(42)
        net = BioMolKGNet(adata.n_vars, mol_dim, kg_dim, hidden_dim=192).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        for _ in range(80):
            net.train()
            opt.zero_grad()
            pred = net(xc_tr, mol_tr, kg_tr, do_tr, dl_tr)
            loss = F.mse_loss(pred, yt) * 50.0 + (1.0 - F.cosine_similarity(pred, yt, dim=-1)).mean() * 0.5
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            ep = net(xc_te, mol_te, kg_te, do_te, dl_te).cpu().numpy()
        df_res = evaluate_perturbation_predictions(ev_true, {f"{s[0]}||{s[1]}": ep[i] for i, s in enumerate(te_s)}, name, (20, 50))
        df_res["dose"] = [int(round(10 ** s[3] - 1)) for s in te_s]
        d10k = df_res[df_res["dose"] >= 9000]
        print(f"  [{name}] All Doses: Pearson={df_res['pearson_delta_all'].mean():.4f} | Top20={df_res['pearson_delta_top20'].mean():.4f} | MSE={df_res['mse_all'].mean():.6f} || 10uM HighDose: Pearson={d10k['pearson_delta_all'].mean():.4f} | Top20={d10k['pearson_delta_top20'].mean():.4f}")
        return df_res

    print("\n=== Unseen Drug Zero-Shot (150 药物训练 -> 38 全新药物测试) ===")
    r_blind = run_kg_model(blind_kg_dict, "BioMolKG-Blind(SMILES->ESM2-Bridge)")
    r_known = run_kg_model(kg_dict, "BioMolKG-TargetKnown(SMILES+ESM2-KG)")
    pd.concat([r_blind, r_known], ignore_index=True).to_csv(r"D:\data\AIVC\test_results\exp5_drug_kg_unseen.csv", index=False, encoding="utf-8")
