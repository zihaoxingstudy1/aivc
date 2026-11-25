# -*- coding: utf-8 -*-
"""
方向 5 (Exp7): 打通 "分子式(SMILES) + 靶点蛋白(ESM-2) -> 单细胞转录组响应(Delta) -> 真实细胞杀伤力(GDSC IC50) -> GTEx 正常器官脱靶毒性" 全商业闭环
基于 McFarlandTsherniak2020 (MIX-seq 209 株泛癌细胞系) + GDSC2/GDSC1 + GTEx v8 (54 正常组织)。
在 Unseen Cell Lines (未见癌细胞系) 严格 OOD 划分下评估转录组 Delta 与 LN_IC50 联合预测能力，并生成正常器官安全性风险热图。
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _to_dense_array


def prepare_mixseq_hvg2000(raw_h5ad: str, out_h5ad: str, n_top_genes: int = 2000):
    if os.path.exists(out_h5ad):
        print(f"[MIX-seq] 加载缓存 HVG2000: {out_h5ad}")
        return sc.read_h5ad(out_h5ad)
    print(f"[MIX-seq] 读取并标准化 {raw_h5ad} ...")
    adata = sc.read_h5ad(raw_h5ad)
    # 仅保留药物扰动与 control
    keep = adata.obs["perturbation_type"].isin(["drug", "control"]) | (adata.obs["perturbation"] == "control")
    # 过滤掉 CRISPR sgRNA 扰动
    is_sg = adata.obs["perturbation"].astype(str).str.startswith("sg")
    adata = adata[keep & (~is_sg)].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    adata.write_h5ad(out_h5ad)
    print(f"[MIX-seq] 保存 HVG2000: {adata.shape}")
    return adata


@torch.no_grad()
def build_mixseq_drug_features(drugs: list, gdsc2_df: pd.DataFrame, cache_npz: str) -> dict:
    if os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=False)
        return {str(k): d["feats"][i] for i, k in enumerate(d["drugs"])}

    # 1. PubChem 查询 SMILES + ChemBERTa + Morgan
    smiles_map = {}
    for dr in drugs:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(dr)}/property/CanonicalSMILES,IsomericSMILES/JSON"
        try:
            res = json.loads(urllib.request.urlopen(url, timeout=10).read().decode("utf-8"))
            props = res["PropertyTable"]["Properties"][0]
            smiles_map[dr] = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES", "")
        except Exception:
            smiles_map[dr] = ""

    print(f"[DrugFeat] PubChem 命中 MIX-seq 药物 SMILES: {sum(bool(v) for v in smiles_map.values())}/{len(drugs)}")

    morgan_list = []
    for dr in drugs:
        sm = smiles_map[dr]
        mol = Chem.MolFromSmiles(sm) if sm else None
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32)
        else:
            fp = np.zeros(1024, dtype=np.float32)
        morgan_list.append(fp)
    morgan_mat = np.vstack(morgan_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb_dir = r"D:\data\AIVC\pretrained\chemberta"
    tok_cb = AutoTokenizer.from_pretrained(cb_dir)
    cb = AutoModel.from_pretrained(cb_dir).to(device).eval()
    cb_list = []
    for dr in drugs:
        sm = smiles_map[dr] or "C"
        enc = tok_cb([sm], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = cb(**enc).last_hidden_state[:, 0, :]
        cb_list.append(F.normalize(out, dim=-1).cpu().numpy()[0])
    cb_mat = np.vstack(cb_list).astype(np.float32)
    del cb

    # 2. ESM-2 靶点嵌入 (来自 GDSC2 PUTATIVE_TARGET)
    tg_map = gdsc2_df.groupby(gdsc2_df["DRUG_NAME"].str.lower())["PUTATIVE_TARGET"].first().to_dict()
    esm_dir = r"D:\data\AIVC\pretrained\esm2_150m"
    tok_e = AutoTokenizer.from_pretrained(esm_dir)
    esm = AutoModel.from_pretrained(esm_dir, torch_dtype=torch.float16).to(device).eval()

    esm_list = []
    for dr in drugs:
        tg = str(tg_map.get(dr.lower(), "")).split(",")[0].strip()
        emb = np.zeros(640, dtype=np.float32)
        if tg and tg != "nan":
            q = f"({tg}) AND organism_id:9606 AND reviewed:true"
            url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({"query": q, "fields": "sequence", "format": "json", "size": 1})
            try:
                res = json.loads(urllib.request.urlopen(url, timeout=10).read().decode("utf-8")).get("results", [])
                if res:
                    seq = res[0].get("sequence", {}).get("value", "")[:1022]
                    if seq:
                        enc = tok_e([seq], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                        out = esm(**enc).last_hidden_state.float()
                        mask = enc["attention_mask"].unsqueeze(-1).float()
                        e = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
                        emb = F.normalize(e, dim=-1).cpu().numpy()[0].astype(np.float32)
            except Exception:
                pass
        esm_list.append(emb)
    esm_mat = np.vstack(esm_list)
    del esm
    torch.cuda.empty_cache()

    feats = np.hstack([morgan_mat, cb_mat, esm_mat]).astype(np.float32)  # 1024 + 384 + 640 = 2048
    np.savez(cache_npz, drugs=np.array(drugs), feats=feats)
    return {dr: feats[i] for i, dr in enumerate(drugs)}


class BioMolKGIC50Net(nn.Module):
    """联合预测单细胞转录组响应 (Delta) 与药敏杀伤力 (LN_IC50, AUC) 的多任务网络。"""
    def __init__(self, n_genes: int, drug_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.delta_head = nn.Linear(hidden_dim, n_genes)
        self.ic50_head = nn.Sequential(
            nn.Linear(hidden_dim + n_genes, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 2),  # [LN_IC50, AUC]
        )

    def forward(self, x_ctrl: torch.Tensor, d_feat: torch.Tensor):
        hc = self.cell_enc(x_ctrl)
        hd = self.drug_enc(d_feat)
        h = self.fuse(torch.cat([hc, hd], dim=-1))
        delta = self.delta_head(h)
        ic50_auc = self.ic50_head(torch.cat([h, delta], dim=-1))
        return delta, ic50_auc


def run_exp7():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = prepare_mixseq_hvg2000(
        r"D:\data\AIVC\data\McFarlandTsherniak2020.h5ad",
        r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad",
    )

    m_list = pd.read_csv(r"D:\data\AIVC\data\cmp_model_list.csv", usecols=["model_id", "BROAD_ID", "model_name"]).dropna(subset=["BROAD_ID"])
    g2 = pd.read_excel(r"D:\data\AIVC\data\GDSC2_fitted_dose_response_27Oct23.xlsx", usecols=["SANGER_MODEL_ID", "DRUG_NAME", "PUTATIVE_TARGET", "LN_IC50", "AUC"])
    g2["drug_l"] = g2["DRUG_NAME"].str.lower()

    # 构建每个 DepMap_ID 的 control 基态表达与 (DepMap_ID, drug) Delta
    obs = adata.obs.copy()
    X_all = _to_dense_array(adata.X).astype(np.float32)

    ctrl_mask = (obs["perturbation"] == "control").values
    cl_ctrl = {}
    for cl, grp in obs[ctrl_mask].groupby("DepMap_ID", observed=True):
        if len(grp) >= 10:
            idx = obs.index.get_indexer(grp.index)
            cl_ctrl[str(cl)] = X_all[idx].mean(axis=0)

    broad_to_sanger = dict(zip(m_list["BROAD_ID"], m_list["model_id"]))
    g2_lookup = g2.groupby(["SANGER_MODEL_ID", "drug_l"])[["LN_IC50", "AUC"]].mean().to_dict("index")

    records = []
    for (cl, dr), grp in obs[~ctrl_mask].groupby(["DepMap_ID", "perturbation"], observed=True):
        cl_s, dr_s = str(cl), str(dr)
        if cl_s not in cl_ctrl or len(grp) < 10:
            continue
        sm_id = broad_to_sanger.get(cl_s)
        if not sm_id:
            continue
        gd = g2_lookup.get((sm_id, dr_s.lower()))
        if not gd:
            continue
        idx = obs.index.get_indexer(grp.index)
        mu_p = X_all[idx].mean(axis=0)
        delta = mu_p - cl_ctrl[cl_s]
        records.append({
            "DepMap_ID": cl_s,
            "drug": dr_s,
            "x_ctrl": cl_ctrl[cl_s],
            "delta": delta,
            "ln_ic50": float(gd["LN_IC50"]),
            "auc": float(gd["AUC"]),
        })

    print(f"[MIX-seq+GDSC2] 有效单细胞转录组+IC50配对样本: {len(records)} 对 ({len({r['DepMap_ID'] for r in records})} 株癌细胞系, {len({r['drug'] for r in records})} 种药物)")

    drugs = sorted({r["drug"] for r in records})
    drug_feats = build_mixseq_drug_features(drugs, g2, r"D:\data\AIVC\data\mixseq_drug_kg_features.npz")

    # 严格 Unseen Cell Line OOD 划分 (80% 细胞系训练 -> 20% 未见癌细胞系测试)
    all_cls = sorted({r["DepMap_ID"] for r in records})
    rng = np.random.default_rng(42)
    rng.shuffle(all_cls)
    n_tr_cl = int(0.8 * len(all_cls))
    tr_cls = set(all_cls[:n_tr_cl])
    te_cls = set(all_cls[n_tr_cl:])

    tr_recs = [r for r in records if r["DepMap_ID"] in tr_cls]
    te_recs = [r for r in records if r["DepMap_ID"] in te_cls]
    print(f"[OOD Split] 训练细胞系 {len(tr_cls)} 株 ({len(tr_recs)} 样本) -> 测试未见细胞系 {len(te_cls)} 株 ({len(te_recs)} 样本)")

    def pack(recs):
        xc = np.vstack([r["x_ctrl"] for r in recs]).astype(np.float32)
        df = np.vstack([drug_feats[r["drug"]] for r in recs]).astype(np.float32)
        yd = np.vstack([r["delta"] for r in recs]).astype(np.float32)
        yi = np.array([[r["ln_ic50"], r["auc"]] for r in recs], dtype=np.float32)
        return xc, df, yd, yi

    tr_xc, tr_df, tr_yd, tr_yi = pack(tr_recs)
    te_xc, te_df, te_yd, te_yi = pack(te_recs)

    # 1. 基线: Ridge (拼接 DrugFeat + CellCtrl) 预测 LN_IC50 & Delta
    X_tr_lin = np.hstack([tr_xc, tr_df])
    X_te_lin = np.hstack([te_xc, te_df])
    ridge_d = Ridge(alpha=100.0).fit(X_tr_lin, tr_yd)
    pred_d_ridge = ridge_d.predict(X_te_lin)
    ridge_i = Ridge(alpha=100.0).fit(X_tr_lin, tr_yi[:, 0])
    pred_i_ridge = ridge_i.predict(X_te_lin)

    # 2. BioMolKG-IC50 联合神经网络
    torch.manual_seed(42)
    net = BioMolKGIC50Net(n_genes=adata.n_vars, drug_dim=tr_df.shape[1], hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    t_xc = torch.tensor(tr_xc, device=device)
    t_df = torch.tensor(tr_df, device=device)
    t_yd = torch.tensor(tr_yd, device=device)
    t_yi = torch.tensor(tr_yi, device=device)

    net.train()
    for ep in range(80):
        perm = torch.randperm(len(tr_recs), device=device)
        for i in range(0, len(tr_recs), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            pd_out, pi_out = net(t_xc[idx], t_df[idx])
            loss_d = F.mse_loss(pd_out, t_yd[idx]) + 0.1 * (1.0 - F.cosine_similarity(pd_out, t_yd[idx], dim=-1).mean())
            loss_i = F.mse_loss(pi_out, t_yi[idx])
            loss = loss_d + 0.05 * loss_i
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    with torch.no_grad():
        pred_d_nn, pred_i_nn = net(torch.tensor(te_xc, device=device), torch.tensor(te_df, device=device))
        pred_d_nn = pred_d_nn.cpu().numpy()
        pred_i_nn = pred_i_nn.cpu().numpy()

    # 评估转录组 Delta 预测
    true_d_dict = {f"{r['DepMap_ID']}||{r['drug']}": r["delta"] for r in te_recs}
    ridge_d_dict = {f"{r['DepMap_ID']}||{r['drug']}": pred_d_ridge[i] for i, r in enumerate(te_recs)}
    nn_d_dict = {f"{r['DepMap_ID']}||{r['drug']}": pred_d_nn[i] for i, r in enumerate(te_recs)}

    df_m1 = evaluate_perturbation_predictions(true_d_dict, ridge_d_dict, "Linear-Ridge(Cell+Drug)")
    df_m2 = evaluate_perturbation_predictions(true_d_dict, nn_d_dict, "BioMolKG-IC50(Joint)")

    # 评估 IC50 & AUC 预测
    r_ic50_ridge, _ = pearsonr(te_yi[:, 0], pred_i_ridge)
    sp_ic50_ridge, _ = spearmanr(te_yi[:, 0], pred_i_ridge)
    r_ic50_nn, _ = pearsonr(te_yi[:, 0], pred_i_nn[:, 0])
    sp_ic50_nn, _ = spearmanr(te_yi[:, 0], pred_i_nn[:, 0])
    r_auc_nn, _ = pearsonr(te_yi[:, 1], pred_i_nn[:, 1])

    print("\n=== Unseen Cancer Cell Lines (OOD) 联合评测 ===")
    print(f"  [转录组 Delta] Linear-Ridge: Pearson={df_m1['pearson_delta_all'].mean():.4f} | Top20={df_m1['pearson_delta_top20'].mean():.4f} | MSE={df_m1['mse_all'].mean():.6f}")
    print(f"  [转录组 Delta] BioMolKG-IC50: Pearson={df_m2['pearson_delta_all'].mean():.4f} | Top20={df_m2['pearson_delta_top20'].mean():.4f} | MSE={df_m2['mse_all'].mean():.6f}")
    print(f"  [药敏 LN_IC50] Linear-Ridge: Pearson={r_ic50_ridge:.4f} | Spearman={sp_ic50_ridge:.4f}")
    print(f"  [药敏 LN_IC50] BioMolKG-IC50: Pearson={r_ic50_nn:.4f} | Spearman={sp_ic50_nn:.4f} | AUC Pearson={r_auc_nn:.4f}")

    # 保存 IC50 测试预测对比
    ic50_df = pd.DataFrame({
        "DepMap_ID": [r["DepMap_ID"] for r in te_recs],
        "drug": [r["drug"] for r in te_recs],
        "true_ln_ic50": te_yi[:, 0],
        "pred_ln_ic50_biomolkg": pred_i_nn[:, 0],
        "pred_ln_ic50_ridge": pred_i_ridge,
        "true_auc": te_yi[:, 1],
        "pred_auc_biomolkg": pred_i_nn[:, 1],
    })
    ic50_df.to_csv(r"D:\data\AIVC\test_results\exp7_unseen_cellline_ic50.csv", index=False, encoding="utf-8")

    # 3. GTEx v8 54 种正常人体器官脱靶毒性与治疗窗风险扫描
    gtex = pd.read_csv(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz", sep="\t", skiprows=2)
    gtex_sym = gtex.drop_duplicates("Description").set_index("Description").iloc[:, 1:]  # [genes, 54 tissues]
    # 对齐到 MIX-seq 2000 HVG 并做 log1p(TPM) 缩放以匹配单细胞尺度
    hvg_names = list(adata.var_names)
    aligned = gtex_sym.reindex(hvg_names).fillna(0.0)
    gtex_x = np.log1p(aligned.values.T.astype(np.float32))  # [54, 2000]
    # 按 MIX-seq control 平均总表达量进行尺度对齐
    ctrl_mean_norm = np.mean(np.linalg.norm(tr_xc, axis=1))
    gtex_norm = np.linalg.norm(gtex_x, axis=1, keepdims=True)
    gtex_x = gtex_x * (ctrl_mean_norm / np.maximum(gtex_norm, 1e-6))

    # 挑选核心安全器官进行展示 + 全部54器官保存
    tissues = list(aligned.columns)
    tox_rows = []
    with torch.no_grad():
        tx = torch.tensor(gtex_x, device=device)
        for dr in drugs:
            td = torch.tensor(np.tile(drug_feats[dr], (len(tissues), 1)), device=device)
            p_delta, p_ic50 = net(tx, td)
            d_mag = torch.norm(p_delta, dim=-1).cpu().numpy()
            ic50_val = p_ic50[:, 0].cpu().numpy()
            for ti, tis in enumerate(tissues):
                tox_rows.append({
                    "drug": dr,
                    "normal_tissue": tis,
                    "predicted_normal_ln_ic50": float(ic50_val[ti]),
                    "predicted_offtarget_delta_norm": float(d_mag[ti]),
                })

    tox_df = pd.DataFrame(tox_rows)
    tox_df.to_csv(r"D:\data\AIVC\test_results\exp7_gtex_normal_organ_toxicity.csv", index=False, encoding="utf-8")

    # 绘制联合图: (左) Unseen Cell Line IC50 散点图; (右) 重点正常器官毒性风险热图 (-LN_IC50 越高杀伤风险越大)
    key_tissues = [
        "Liver", "Heart - Left Ventricle", "Kidney - Cortex", "Lung",
        "Brain - Cortex", "Whole Blood", "Colon - Transverse", "Pancreas", "Skin - Sun Exposed (Lower leg)"
    ]
    sub_tox = tox_df[tox_df["normal_tissue"].isin(key_tissues)].pivot(index="normal_tissue", columns="drug", values="predicted_normal_ln_ic50")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    sns.regplot(data=ic50_df, x="true_ln_ic50", y="pred_ln_ic50_biomolkg", ax=axes[0], scatter_kws={"s": 40, "alpha": 0.75, "color": "#2980b9"}, line_kws={"color": "#e74c3c"})
    axes[0].set_title(f"Unseen Cancer Cell Lines LN_IC50 Prediction\nPearson r = {r_ic50_nn:.3f} | Spearman = {sp_ic50_nn:.3f}", fontweight="bold")
    axes[0].set_xlabel("True GDSC2 LN_IC50")
    axes[0].set_ylabel("Predicted LN_IC50 (BioMolKG-IC50)")

    sns.heatmap(sub_tox, cmap="RdYlGn", center=0.0, annot=True, fmt=".2f", ax=axes[1], cbar_kws={"label": "Predicted Normal Tissue LN_IC50 (Lower = Higher Risk)"})
    axes[1].set_title("GTEx Normal Organ Off-Target Toxicity Panel", fontweight="bold")
    axes[1].set_xlabel("Drug")
    axes[1].set_ylabel("Normal Human Organ (GTEx v8)")

    plt.tight_layout()
    plt.savefig(r"D:\data\AIVC\test_results\exp7_ic50_and_gtex_toxicity.png", bbox_inches="tight")
    plt.close(fig)
    print("[Exp7] 结果已保存至 exp7_unseen_cellline_ic50.csv, exp7_gtex_normal_organ_toxicity.csv, exp7_ic50_and_gtex_toxicity.png")


if __name__ == "__main__":
    run_exp7()
