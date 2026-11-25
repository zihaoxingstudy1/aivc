# -*- coding: utf-8 -*-
"""
方向 6 (Exp10): 接入 DrugComb v1.5 + Sanger GDSC Combinations 真实湿实验双药协同数据
构建对称集合交互网络 BioMolKG-Synergy (Symmetric Drug-Pair Set-Interaction + Cell Baseline + ESM-2 Target KG)
在严格 Unseen Cell Lines 与 Unseen Drug Pairs 双维度 OOD 划分下评测双药协同 (Bliss / ZIP / GDSC Synergy ROC-AUC)。
"""

import os
import sys
import glob
import re
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
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


def load_ccle_sanger_expr_by_id_and_name(tpm_csv: str, model_csv: str, hvg_genes: list):
    print(f"[Expr] 读取 CCLE/Sanger 表达矩阵: {tpm_csv}")
    df = pd.read_csv(tpm_csv, skiprows=[1, 2, 3], low_memory=False)
    df = df.rename(columns={"model_id": "gene_symbol"}).drop(columns=["Unnamed: 1", "Unnamed: 2"], errors="ignore")
    df = df.dropna(subset=["gene_symbol"]).drop_duplicates("gene_symbol").set_index("gene_symbol")
    aligned = df.reindex(hvg_genes).fillna(0.0).astype(np.float32)
    mat = np.log1p(np.clip(aligned.values.T, 0.0, None))
    mu = mat.mean(axis=1, keepdims=True)
    sd = np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
    mat = (mat - mu) / sd
    sidms = list(aligned.columns)
    sidm_to_vec = {m: mat[i] for i, m in enumerate(sidms)}

    ml = pd.read_csv(model_csv, usecols=["model_id", "model_name", "synonyms"]).dropna(subset=["model_id"])
    name_to_sidm = {}
    for _, r in ml.iterrows():
        sidm = str(r["model_id"])
        if sidm not in sidm_to_vec:
            continue
        nm = str(r["model_name"]).strip().upper()
        name_to_sidm[nm] = sidm
        name_to_sidm[re.sub(r"[^A-Z0-9]", "", nm)] = sidm
    return sidm_to_vec, name_to_sidm, mat.shape[1]


def _fetch_pubchem_smiles(drug_name: str) -> tuple:
    clean = re.sub(r"\s*\(.*?\)", "", str(drug_name)).strip()
    for q in (clean, str(drug_name).strip()):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(q)}/property/CanonicalSMILES,IsomericSMILES/JSON"
        try:
            res = json.loads(urllib.request.urlopen(url, timeout=6).read().decode("utf-8"))
            props = res["PropertyTable"]["Properties"][0]
            sm = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES", "")
            if sm:
                return drug_name, sm
        except Exception:
            continue
    return drug_name, ""


@torch.no_grad()
def build_combo_drug_features(drugs: list, gdsc_npz: str, cache_npz: str) -> dict:
    # 先复用已缓存的 gdsc_drug_kg_features.npz
    cached = {}
    if os.path.exists(gdsc_npz):
        d = np.load(gdsc_npz, allow_pickle=False)
        for i, k in enumerate(d["drugs"]):
            cached[str(k).lower()] = d["feats"][i]
    if os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=False)
        for i, k in enumerate(d["drugs"]):
            cached[str(k).lower()] = d["feats"][i]

    missing = [dr for dr in drugs if dr.lower() not in cached]
    print(f"[DrugFeat] 总药物 {len(drugs)}, 命中本地缓存 {len(drugs) - len(missing)}, 需编码 {len(missing)}")

    if missing:
        smiles_map = {}
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(_fetch_pubchem_smiles, dr) for dr in missing]
            for fut in as_completed(futs):
                dr, sm = fut.result()
                smiles_map[dr] = sm
        print(f"[DrugFeat] PubChem 补全 SMILES: {sum(bool(v) for v in smiles_map.values())}/{len(missing)}")

        morgan_list = []
        for dr in missing:
            sm = smiles_map.get(dr, "")
            mol = Chem.MolFromSmiles(sm) if sm else None
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32) if mol is not None else np.zeros(1024, dtype=np.float32)
            morgan_list.append(fp)
        morgan_mat = np.vstack(morgan_list)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cb_dir = r"D:\data\AIVC\pretrained\chemberta"
        tok_cb = AutoTokenizer.from_pretrained(cb_dir)
        cb = AutoModel.from_pretrained(cb_dir).to(device).eval()
        cb_list = []
        for i in range(0, len(missing), 32):
            batch_sm = [smiles_map.get(dr, "") or "C" for dr in missing[i : i + 32]]
            enc = tok_cb(batch_sm, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = cb(**enc).last_hidden_state[:, 0, :]
            cb_list.append(F.normalize(out, dim=-1).cpu().numpy().astype(np.float32))
        cb_mat = np.vstack(cb_list)
        del cb
        torch.cuda.empty_cache()

        esm_pad = np.zeros((len(missing), 640), dtype=np.float32)
        new_feats = np.hstack([morgan_mat, cb_mat, esm_pad]).astype(np.float32)
        for i, dr in enumerate(missing):
            cached[dr.lower()] = new_feats[i]

    # 对未命中 SMILES 的药物，加上可学习的哈希指纹兜底特征 (前 1024 维若全零则用名称确定性随机投影，确保不同药物有可区分表示但无泄漏)
    out_dict = {}
    for dr in drugs:
        vec = cached.get(dr.lower(), np.zeros(2048, dtype=np.float32)).copy()
        if np.sum(np.abs(vec[:1024])) == 0:
            seed = abs(hash(dr.lower())) % (2**31)
            rng = np.random.default_rng(seed)
            vec[:256] = rng.normal(0, 0.1, size=256).astype(np.float32)
        out_dict[dr] = vec

    np.savez(cache_npz, drugs=np.array(drugs), feats=np.vstack([out_dict[d] for d in drugs]).astype(np.float32))
    return out_dict


class SymmetricComboSynergyNet(nn.Module):
    """对称双药联合协同预测网络: f(Cell, A, B) == f(Cell, B, A)。"""
    def __init__(self, n_genes: int, drug_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # 对称组合特征: [hc, ha+hb, ha*hb, |ha-hb|] -> 4 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, xc: torch.Tensor, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        hc = self.cell_enc(xc)
        ha = self.drug_enc(xa)
        hb = self.drug_enc(xb)
        sym = torch.cat([hc, ha + hb, ha * hb, torch.abs(ha - hb)], dim=-1)
        return self.head(sym).squeeze(-1)


def eval_regression_task(name: str, df: pd.DataFrame, cell_dict: dict, drug_dict: dict, n_genes: int, drug_dim: int, target_col: str, device):
    rng = np.random.default_rng(42)
    results = []
    for split_type in ("UnseenCellLines", "UnseenDrugPairs"):
        if split_type == "UnseenCellLines":
            cls = sorted(df["SIDM"].unique())
            rng.shuffle(cls)
            tr_set = set(cls[: int(0.8 * len(cls))])
            tr = df[df["SIDM"].isin(tr_set)]
            te = df[~df["SIDM"].isin(tr_set)]
        else:
            pairs = sorted(df["pair_key"].unique())
            rng.shuffle(pairs)
            tr_set = set(pairs[: int(0.8 * len(pairs))])
            tr = df[df["pair_key"].isin(tr_set)]
            te = df[~df["pair_key"].isin(tr_set)]

        def pack(sub):
            xc = np.vstack([cell_dict[m] for m in sub["SIDM"]]).astype(np.float32)
            xa = np.vstack([drug_dict[d] for d in sub["drug_a"]]).astype(np.float32)
            xb = np.vstack([drug_dict[d] for d in sub["drug_b"]]).astype(np.float32)
            y = sub[target_col].values.astype(np.float32)
            return xc, xa, xb, y

        tr_xc, tr_xa, tr_xb, tr_y = pack(tr)
        te_xc, te_xa, te_xb, te_y = pack(te)

        # 对称 Ridge 基线: [xc, xa+xb, xa*xb]
        X_tr_lin = np.hstack([tr_xc, tr_xa + tr_xb, tr_xa * tr_xb])
        X_te_lin = np.hstack([te_xc, te_xa + te_xb, te_xa * te_xb])
        ridge = Ridge(alpha=500.0).fit(X_tr_lin, tr_y)
        pred_r = ridge.predict(X_te_lin)
        r_ridge, _ = pearsonr(te_y, pred_r)
        sp_ridge, _ = spearmanr(te_y, pred_r)

        torch.manual_seed(42)
        net = SymmetricComboSynergyNet(n_genes, drug_dim, hidden_dim=256).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

        t_xc = torch.tensor(tr_xc, device=device)
        t_xa = torch.tensor(tr_xa, device=device)
        t_xb = torch.tensor(tr_xb, device=device)
        t_y = torch.tensor(tr_y, device=device)

        net.train()
        for ep in range(20):
            perm = torch.randperm(len(tr_y), device=device)
            for i in range(0, len(tr_y), 1024):
                idx = perm[i : i + 1024]
                opt.zero_grad()
                out = net(t_xc[idx], t_xa[idx], t_xb[idx])
                loss = F.mse_loss(out, t_y[idx])
                loss.backward()
                opt.step()
            sched.step()

        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(te_y), 2048):
                preds.append(net(
                    torch.tensor(te_xc[i : i + 2048], device=device),
                    torch.tensor(te_xa[i : i + 2048], device=device),
                    torch.tensor(te_xb[i : i + 2048], device=device),
                ).cpu().numpy())
        pred_nn = np.concatenate(preds)
        r_nn, _ = pearsonr(te_y, pred_nn)
        sp_nn, _ = spearmanr(te_y, pred_nn)

        print(f"  [{name} | {split_type}] Train={len(tr_y)} Test={len(te_y)}")
        print(f"    Ridge:            Pearson={r_ridge:.4f} | Spearman={sp_ridge:.4f}")
        print(f"    BioMolKG-Synergy: Pearson={r_nn:.4f} | Spearman={sp_nn:.4f}")
        results.append({
            "dataset": name, "split": split_type, "metric": target_col,
            "n_train": len(tr_y), "n_test": len(te_y),
            "ridge_pearson": r_ridge, "ridge_spearman": sp_ridge,
            "nn_pearson": r_nn, "nn_spearman": sp_nn,
        })
    return results


def eval_gdsc_synergy_cls(df: pd.DataFrame, cell_dict: dict, drug_dict: dict, n_genes: int, drug_dim: int, device):
    rng = np.random.default_rng(42)
    results = []
    for split_type in ("UnseenCellLines", "UnseenDrugPairs"):
        if split_type == "UnseenCellLines":
            cls = sorted(df["SIDM"].unique())
            rng.shuffle(cls)
            tr_set = set(cls[: int(0.8 * len(cls))])
            tr = df[df["SIDM"].isin(tr_set)]
            te = df[~df["SIDM"].isin(tr_set)]
        else:
            pairs = sorted(df["pair_key"].unique())
            rng.shuffle(pairs)
            tr_set = set(pairs[: int(0.8 * len(pairs))])
            tr = df[df["pair_key"].isin(tr_set)]
            te = df[~df["pair_key"].isin(tr_set)]

        def pack(sub):
            xc = np.vstack([cell_dict[m] for m in sub["SIDM"]]).astype(np.float32)
            xa = np.vstack([drug_dict[d] for d in sub["drug_a"]]).astype(np.float32)
            xb = np.vstack([drug_dict[d] for d in sub["drug_b"]]).astype(np.float32)
            y = sub["Synergy?"].values.astype(np.float32)
            return xc, xa, xb, y

        tr_xc, tr_xa, tr_xb, tr_y = pack(tr)
        te_xc, te_xa, te_xb, te_y = pack(te)

        # Ridge 分类基线
        ridge = Ridge(alpha=500.0).fit(np.hstack([tr_xc, tr_xa + tr_xb, tr_xa * tr_xb]), tr_y)
        pred_r = ridge.predict(np.hstack([te_xc, te_xa + te_xb, te_xa * te_xb]))
        auc_r = roc_auc_score(te_y, pred_r)
        ap_r = average_precision_score(te_y, pred_r)

        torch.manual_seed(42)
        net = SymmetricComboSynergyNet(n_genes, drug_dim, hidden_dim=256).to(device)
        pos_w = torch.tensor([(len(tr_y) - tr_y.sum()) / max(tr_y.sum(), 1.0)], device=device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

        t_xc = torch.tensor(tr_xc, device=device)
        t_xa = torch.tensor(tr_xa, device=device)
        t_xb = torch.tensor(tr_xb, device=device)
        t_y = torch.tensor(tr_y, device=device)

        net.train()
        for ep in range(20):
            perm = torch.randperm(len(tr_y), device=device)
            for i in range(0, len(tr_y), 1024):
                idx = perm[i : i + 1024]
                opt.zero_grad()
                logits = net(t_xc[idx], t_xa[idx], t_xb[idx])
                loss = F.binary_cross_entropy_with_logits(logits, t_y[idx], pos_weight=pos_w)
                loss.backward()
                opt.step()
            sched.step()

        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(te_y), 2048):
                preds.append(torch.sigmoid(net(
                    torch.tensor(te_xc[i : i + 2048], device=device),
                    torch.tensor(te_xa[i : i + 2048], device=device),
                    torch.tensor(te_xb[i : i + 2048], device=device),
                )).cpu().numpy())
        pred_nn = np.concatenate(preds)
        auc_nn = roc_auc_score(te_y, pred_nn)
        ap_nn = average_precision_score(te_y, pred_nn)

        print(f"  [GDSC_Anchor_Synergy | {split_type}] Train={len(tr_y)} (pos={int(tr_y.sum())}) Test={len(te_y)} (pos={int(te_y.sum())})")
        print(f"    Ridge:            ROC-AUC={auc_r:.4f} | PR-AUC={ap_r:.4f}")
        print(f"    BioMolKG-Synergy: ROC-AUC={auc_nn:.4f} | PR-AUC={ap_nn:.4f}")
        results.append({
            "dataset": "GDSC_Anchor_Combos", "split": split_type, "metric": "Synergy?_ROC_AUC",
            "n_train": len(tr_y), "n_test": len(te_y),
            "ridge_pearson": auc_r, "ridge_spearman": ap_r,
            "nn_pearson": auc_nn, "nn_spearman": ap_nn,
        })
    return results


def run_exp10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)

    sidm_to_vec, name_to_sidm, n_genes = load_ccle_sanger_expr_by_id_and_name(
        r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv",
        r"D:\data\AIVC\data\cmp_model_list.csv",
        hvg_genes,
    )

    # 1. 加载 DrugComb v1.5 真实双药组合
    print("[DrugComb v1.5] 读取真实双药协同数据 ...")
    dc = pd.read_csv(
        r"D:\data\drug\summary_v_1_5.csv",
        usecols=["drug_row", "drug_col", "cell_line_name", "synergy_zip", "synergy_bliss"],
        low_memory=False,
    ).dropna(subset=["drug_row", "drug_col", "cell_line_name", "synergy_bliss"])
    dc["SIDM"] = dc["cell_line_name"].str.strip().str.upper().map(name_to_sidm)
    dc = dc.dropna(subset=["SIDM"]).copy()
    dc["drug_a"] = dc[["drug_row", "drug_col"]].min(axis=1).astype(str)
    dc["drug_b"] = dc[["drug_row", "drug_col"]].max(axis=1).astype(str)
    dc["pair_key"] = dc["drug_a"] + "||" + dc["drug_b"]
    # 聚合相同 (SIDM, drug_a, drug_b) 的实验重复均值
    dc_agg = dc.groupby(["SIDM", "drug_a", "drug_b", "pair_key"], as_index=False)[["synergy_bliss", "synergy_zip"]].mean()
    # 过滤极端异常协同值 ([-100, 100])
    dc_agg = dc_agg[(dc_agg["synergy_bliss"].between(-100, 100)) & (dc_agg["synergy_zip"].between(-100, 100))].reset_index(drop=True)
    print(f"[DrugComb v1.5] 去重后真实双药组合样本: {len(dc_agg)} 对 ({dc_agg['SIDM'].nunique()} 株细胞系, {dc_agg['pair_key'].nunique()} 种药物组合)")

    # 2. 加载 Sanger GDSC Combinations (Anchor 3 文件)
    anchor_files = sorted(glob.glob(r"D:\data\drug\GDSC2drug\*_anchor_combo.csv.gz"))
    g_list = []
    for f in anchor_files:
        sub = pd.read_csv(f, usecols=["SDIM", "Anchor Name", "Library Name", "Delta Emax", "Synergy?"])
        g_list.append(sub)
    gdsc_anc = pd.concat(g_list, ignore_index=True).dropna(subset=["SDIM", "Anchor Name", "Library Name", "Synergy?"])
    gdsc_anc = gdsc_anc[gdsc_anc["SDIM"].isin(sidm_to_vec)].copy()
    gdsc_anc["SIDM"] = gdsc_anc["SDIM"].astype(str)
    gdsc_anc["drug_a"] = gdsc_anc[["Anchor Name", "Library Name"]].min(axis=1).astype(str)
    gdsc_anc["drug_b"] = gdsc_anc[["Anchor Name", "Library Name"]].max(axis=1).astype(str)
    gdsc_anc["pair_key"] = gdsc_anc["drug_a"] + "||" + gdsc_anc["drug_b"]
    gdsc_anc = gdsc_anc.groupby(["SIDM", "drug_a", "drug_b", "pair_key"], as_index=False)[["Delta Emax", "Synergy?"]].max()
    print(f"[Sanger GDSC Combos] 去重后真实双药组合样本: {len(gdsc_anc)} 对 ({gdsc_anc['SIDM'].nunique()} 株细胞系, {gdsc_anc['pair_key'].nunique()} 种组合, 正协同率={gdsc_anc['Synergy?'].mean():.2%})")

    all_drugs = sorted(set(dc_agg["drug_a"]) | set(dc_agg["drug_b"]) | set(gdsc_anc["drug_a"]) | set(gdsc_anc["drug_b"]))
    drug_dict = build_combo_drug_features(
        all_drugs,
        r"D:\data\AIVC\data\gdsc_drug_kg_features.npz",
        r"D:\data\AIVC\data\combo_drug_kg_features.npz",
    )
    drug_dim = len(next(iter(drug_dict.values())))

    all_res = []
    print("\n=== Benchmark 1: DrugComb v1.5 (Bliss Synergy Regression) ===")
    all_res.extend(eval_regression_task("DrugComb_v1.5", dc_agg, sidm_to_vec, drug_dict, n_genes, drug_dim, "synergy_bliss", device))

    print("\n=== Benchmark 2: Sanger GDSC Combinations (Synergy Classification ROC-AUC / PR-AUC) ===")
    all_res.extend(eval_gdsc_synergy_cls(gdsc_anc, sidm_to_vec, drug_dict, n_genes, drug_dim, device))

    res_df = pd.DataFrame(all_res)
    res_df.to_csv(r"D:\data\AIVC\test_results\exp10_drug_combo_synergy.csv", index=False, encoding="utf-8")
    print("\n[Exp10] 结果已保存至 exp10_drug_combo_synergy.csv")


if __name__ == "__main__":
    run_exp10()
