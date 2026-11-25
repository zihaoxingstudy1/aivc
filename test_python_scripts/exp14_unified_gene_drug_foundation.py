# -*- coding: utf-8 -*-
"""
思路 A (Exp14): 基因-药物统一合流大模型 (Unified Perturbation-Drug Foundation via ESM-2 Target Space)
核心验证:
1. 阶段 1 (Gene Epistasis Pretraining): 在单细胞双基因/多基因扰动 (Norman2019 + Tian2019 + Wessels2023) 的 ESM-2 靶点空间上预训练对称交互编码器，学习“两个蛋白靶点同时被干预时的非线性协同规律”。
2. 阶段 2 (Zero-Shot / Few-Shot Transfer to Drug Synergy): 迁移到 Sanger GDSC Combinations 真实双药组合协同 (Unseen Drug Pairs)，对比“无基因预训练” vs “基因上位性预训练初始化 (ESM-2 Bridge)”，在 10% 低样本与 100% 全量下的 ROC-AUC / PR-AUC 增益！
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
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def _fetch_uniprot(gene_sym: str) -> tuple:
    g = re.sub(r"[^A-Za-z0-9\-]", "", str(gene_sym).split(",")[0].strip())
    if not g or g.lower() in ("nan", "control", "none"):
        return gene_sym, ""
    q = f"(gene:{g} OR {g}) AND organism_id:9606 AND reviewed:true"
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({"query": q, "fields": "sequence", "format": "json", "size": 1})
    try:
        res = json.loads(urllib.request.urlopen(url, timeout=6).read().decode("utf-8")).get("results", [])
        if res:
            return gene_sym, res[0].get("sequence", {}).get("value", "")[:1022]
    except Exception:
        pass
    return gene_sym, ""


@torch.no_grad()
def build_unified_esm2_dict(genes: list, cache_npz: str) -> dict:
    cached = {}
    for p in (r"D:\data\AIVC\data\target_esm2_150m.npz", cache_npz):
        if os.path.exists(p):
            d = np.load(p, allow_pickle=False)
            kname = "genes" if "genes" in d else "targets"
            vname = "embs" if "embs" in d else "target_embs"
            if kname in d and vname in d:
                for i, k in enumerate(d[kname]):
                    if np.linalg.norm(d[vname][i]) > 1e-6:
                        cached[str(k).upper()] = d[vname][i]

    missing = sorted({g.upper() for g in genes if g.upper() not in cached})
    print(f"[Unified-ESM2] 靶点总数 {len(genes)}, 命中缓存 {len(genes) - len(missing)}, 需检索 {len(missing)}")
    if missing:
        seqs = {}
        with ThreadPoolExecutor(max_workers=12) as ex:
            for g, s in ex.map(_fetch_uniprot, missing):
                if s:
                    seqs[g] = s
        print(f"[Unified-ESM2] UniProt 补全序列: {len(seqs)}/{len(missing)}")
        if seqs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            esm_dir = r"D:\data\AIVC\pretrained\esm2_150m"
            tok = AutoTokenizer.from_pretrained(esm_dir)
            esm = AutoModel.from_pretrained(esm_dir, torch_dtype=torch.float16).to(device).eval()
            keys = sorted(seqs.keys())
            for i in range(0, len(keys), 16):
                enc = tok([seqs[k] for k in keys[i : i + 16]], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                out = esm(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                e = F.normalize((out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6), dim=-1).cpu().numpy().astype(np.float32)
                for j, k in enumerate(keys[i : i + 16]):
                    cached[k] = e[j]
            del esm
            torch.cuda.empty_cache()

    out_dict = {g.upper(): cached.get(g.upper(), np.zeros(640, dtype=np.float32)) for g in genes}
    all_k = sorted(out_dict.keys())
    np.savez(cache_npz, genes=np.array(all_k), embs=np.vstack([out_dict[k] for k in all_k]).astype(np.float32))
    return out_dict


class UnifiedTargetInteractionFoundation(nn.Module):
    """统一基因-药物靶点交互大模型：通过 ESM-2 靶点空间共享主干。"""
    def __init__(self, n_genes: int, esm_dim: int = 640, drug_mol_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.target_enc = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # 共享的对称双靶点交互骨干 (由基因上位性数据预训练)
        self.pair_backbone = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.gene_epistasis_head = nn.Linear(hidden_dim, 1)

        # 下游药物-细胞融合头
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.mol_res = nn.Sequential(
            nn.Linear(drug_mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.drug_synergy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(192, 1),
        )

    def encode_target_pair(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        ha = self.target_enc(ea)
        hb = self.target_enc(eb)
        return self.pair_backbone(torch.cat([ha + hb, ha * hb, torch.abs(ha - hb)], dim=-1))

    def forward_gene_epistasis(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        hp = self.encode_target_pair(ea, eb)
        return self.gene_epistasis_head(hp).squeeze(-1)

    def forward_drug_synergy(self, xc: torch.Tensor, ea: torch.Tensor, eb: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
        hp = self.encode_target_pair(ea, eb)
        hc = self.cell_enc(xc)
        hma = self.mol_res(ma)
        hmb = self.mol_res(mb)
        hm = hma + hmb + (hma * hmb)
        return self.drug_synergy_head(torch.cat([hc, hp, hm], dim=-1)).squeeze(-1)


def run_exp14():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载基因双靶点协同数据 (来自 Exp12b Norman + iPSC 机制表)
    n_df = pd.read_csv(r"D:\data\AIVC\test_results\exp12b_norman_mechanism.csv")
    i_df = pd.read_csv(r"D:\data\AIVC\test_results\exp12b_ipsc_mechanism.csv")
    g_comb = pd.concat([n_df, i_df], ignore_index=True)
    g_comb = g_comb[g_comb["k_genes"] == 2].copy()
    g_comb["ga"] = g_comb["combo"].str.split("_").str[0].str.upper()
    g_comb["gb"] = g_comb["combo"].str.split("_").str[1].str.upper()
    # 协同目标: epistasis_gain_l2 (标准化)
    y_epi = g_comb["epistasis_gain_l2"].values.astype(np.float32)
    y_epi = (y_epi - y_epi.mean()) / (y_epi.std() + 1e-6)

    # 2. 加载 Sanger GDSC Combinations (Anchor 3 文件，带精确靶点名)
    anchor_files = sorted(glob.glob(r"D:\data\drug\GDSC2drug\*_anchor_combo.csv.gz"))
    g_list = [pd.read_csv(f, usecols=["SDIM", "Anchor Name", "Anchor Target", "Library Name", "library Target", "Synergy?"]) for f in anchor_files]
    gdsc_anc = pd.concat(g_list, ignore_index=True).dropna(subset=["SDIM", "Anchor Name", "Library Name", "Synergy?"])

    # 加载细胞基态 & 药物特征
    ref = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
    hvg_genes = list(ref.var_names)
    tpm = pd.read_csv(r"D:\data\AIVC\data\rnaseq_merged_rsem_tpm_20260323.csv", skiprows=[1, 2, 3], low_memory=False)
    tpm = tpm.rename(columns={"model_id": "gene_symbol"}).drop(columns=["Unnamed: 1", "Unnamed: 2"], errors="ignore")
    tpm = tpm.dropna(subset=["gene_symbol"]).drop_duplicates("gene_symbol").set_index("gene_symbol").reindex(hvg_genes).fillna(0.0).astype(np.float32)
    mat = np.log1p(np.clip(tpm.values.T, 0.0, None))
    mat = (mat - mat.mean(axis=1, keepdims=True)) / np.maximum(mat.std(axis=1, keepdims=True), 1e-6)
    sidm_to_vec = {m: mat[i] for i, m in enumerate(tpm.columns)}

    gdsc_anc = gdsc_anc[gdsc_anc["SDIM"].isin(sidm_to_vec)].copy()
    gdsc_anc["drug_a"] = gdsc_anc[["Anchor Name", "Library Name"]].min(axis=1).astype(str)
    gdsc_anc["drug_b"] = gdsc_anc[["Anchor Name", "Library Name"]].max(axis=1).astype(str)
    gdsc_anc["pair_key"] = gdsc_anc["drug_a"] + "||" + gdsc_anc["drug_b"]
    # 保留每个药物的第一个靶点关键词
    d_to_tg = {}
    for _, r in gdsc_anc[["Anchor Name", "Anchor Target"]].drop_duplicates().iterrows():
        d_to_tg[str(r["Anchor Name"])] = str(r["Anchor Target"]).split(",")[0].strip().upper()
    for _, r in gdsc_anc[["Library Name", "library Target"]].drop_duplicates().iterrows():
        if str(r["Library Name"]) not in d_to_tg:
            d_to_tg[str(r["Library Name"])] = str(r["library Target"]).split(",")[0].strip().upper()

    gdsc_anc = gdsc_anc.groupby(["SDIM", "drug_a", "drug_b", "pair_key"], as_index=False)["Synergy?"].max()

    cd = np.load(r"D:\data\AIVC\data\combo_drug_kg_features.npz", allow_pickle=False)
    mol_dict = {str(k).lower(): cd["feats"][i] for i, k in enumerate(cd["drugs"])}

    # 构建统一的 ESM-2 靶点语义字典 (同时覆盖基因扰动靶点 + 药物蛋白靶点)
    all_target_genes = sorted(set(g_comb["ga"]) | set(g_comb["gb"]) | set(d_to_tg.values()))
    esm2_dict = build_unified_esm2_dict(all_target_genes, r"D:\data\AIVC\data\unified_esm2_targets.npz")

    # 准备基因上位性预训练张量
    g_ea = torch.tensor(np.vstack([esm2_dict[g] for g in g_comb["ga"]]), device=device)
    g_eb = torch.tensor(np.vstack([esm2_dict[g] for g in g_comb["gb"]]), device=device)
    g_y = torch.tensor(y_epi, device=device)

    # 划分 GDSC Unseen Drug Pairs (80% 组合训练 / 20% 未见药物组合测试)
    pairs = sorted(gdsc_anc["pair_key"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(pairs)
    tr_pairs = set(pairs[: int(0.8 * len(pairs))])
    tr_full = gdsc_anc[gdsc_anc["pair_key"].isin(tr_pairs)].reset_index(drop=True)
    te_df = gdsc_anc[~gdsc_anc["pair_key"].isin(tr_pairs)].reset_index(drop=True)

    def pack_drug(sub):
        xc = np.vstack([sidm_to_vec[m] for m in sub["SDIM"]]).astype(np.float32)
        ea = np.vstack([esm2_dict.get(d_to_tg.get(d, ""), np.zeros(640, dtype=np.float32)) for d in sub["drug_a"]]).astype(np.float32)
        eb = np.vstack([esm2_dict.get(d_to_tg.get(d, ""), np.zeros(640, dtype=np.float32)) for d in sub["drug_b"]]).astype(np.float32)
        ma = np.vstack([mol_dict.get(d.lower(), np.zeros(2048, dtype=np.float32)) for d in sub["drug_a"]]).astype(np.float32)
        mb = np.vstack([mol_dict.get(d.lower(), np.zeros(2048, dtype=np.float32)) for d in sub["drug_b"]]).astype(np.float32)
        y = sub["Synergy?"].values.astype(np.float32)
        return xc, ea, eb, ma, mb, y

    te_xc, te_ea, te_eb, te_ma, te_mb, te_y = pack_drug(te_df)

    results = []
    for frac in (0.10, 1.0):
        sub_tr = tr_full.sample(frac=frac, random_state=42).reset_index(drop=True) if frac < 1.0 else tr_full
        tr_xc, tr_ea, tr_eb, tr_ma, tr_mb, tr_y = pack_drug(sub_tr)

        for pretrain_gene, label in [(False, "Drug-Only(NoGenePretrain)"), (True, "Unified-Gene2Drug(ESM2-Epistasis)")]:
            torch.manual_seed(42)
            net = UnifiedTargetInteractionFoundation(n_genes=len(hvg_genes)).to(device)

            if pretrain_gene:
                opt_g = torch.optim.AdamW(list(net.target_enc.parameters()) + list(net.pair_backbone.parameters()) + list(net.gene_epistasis_head.parameters()), lr=2e-3, weight_decay=1e-4)
                net.train()
                for _ in range(120):
                    opt_g.zero_grad()
                    loss_g = F.mse_loss(net.forward_gene_epistasis(g_ea, g_eb), g_y)
                    loss_g.backward()
                    opt_g.step()

            pos_w = torch.tensor([(len(tr_y) - tr_y.sum()) / max(tr_y.sum(), 1.0)], device=device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

            t_xc = torch.tensor(tr_xc, device=device)
            t_ea = torch.tensor(tr_ea, device=device)
            t_eb = torch.tensor(tr_eb, device=device)
            t_ma = torch.tensor(tr_ma, device=device)
            t_mb = torch.tensor(tr_mb, device=device)
            t_y = torch.tensor(tr_y, device=device)

            net.train()
            for ep in range(20):
                perm = torch.randperm(len(tr_y), device=device)
                for i in range(0, len(tr_y), 1024):
                    idx = perm[i : i + 1024]
                    opt.zero_grad()
                    # 多任务防遗忘: 主任务药物协同 + 0.05 基因上位性正则
                    logits = net.forward_drug_synergy(t_xc[idx], t_ea[idx], t_eb[idx], t_ma[idx], t_mb[idx])
                    loss = F.binary_cross_entropy_with_logits(logits, t_y[idx], pos_weight=pos_w)
                    if pretrain_gene:
                        loss = loss + 0.05 * F.mse_loss(net.forward_gene_epistasis(g_ea, g_eb), g_y)
                    loss.backward()
                    opt.step()
                sched.step()

            net.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(te_y), 2048):
                    preds.append(torch.sigmoid(net.forward_drug_synergy(
                        torch.tensor(te_xc[i : i + 2048], device=device),
                        torch.tensor(te_ea[i : i + 2048], device=device),
                        torch.tensor(te_eb[i : i + 2048], device=device),
                        torch.tensor(te_ma[i : i + 2048], device=device),
                        torch.tensor(te_mb[i : i + 2048], device=device),
                    )).cpu().numpy())
            pred = np.concatenate(preds)
            auc = roc_auc_score(te_y, pred)
            ap = average_precision_score(te_y, pred)
            print(f"  [TrainFrac={int(frac*100)}% | {label}] UnseenDrugPairs ROC-AUC={auc:.4f} | PR-AUC={ap:.4f}")
            results.append({"train_frac": frac, "model": label, "roc_auc": auc, "pr_auc": ap})

    pd.DataFrame(results).to_csv(r"D:\data\AIVC\test_results\exp14_unified_gene2drug_transfer.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    run_exp14()
