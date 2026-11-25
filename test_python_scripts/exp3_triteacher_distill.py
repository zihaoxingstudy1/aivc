# -*- coding: utf-8 -*-
"""
方向 3 实验：多教师集成蒸馏横向评测 (Geneformer-V2-316M + tdc/scGPT + ESM2-150M)
在 Norman 2019 与 Shifrut 2018 (原代 T 细胞) 上对比单教师与三教师集成蒸馏表现。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
from safetensors import safe_open
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "func_python_scripts"))
from lbb import evaluate_perturbation_predictions, _compute_group_deltas, _to_dense_array
from gf_distill import extract_geneformer_gene_embeddings


def load_scgpt_gene_embeddings(vocab_json: str, safetensors_path: str, target_genes: list) -> dict:
    with open(vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with safe_open(safetensors_path, framework="pt") as sf:
        emb_w = sf.get_tensor("gene_encoder.embedding.weight").float().numpy()

    out = {}
    for g in target_genes:
        tid = vocab.get(g) or vocab.get(g.upper())
        if tid is not None and int(tid) < emb_w.shape[0]:
            v = emb_w[int(tid)].copy()
            out[g] = v / (np.linalg.norm(v) + 1e-6)
    print(f"[scGPT] 成功匹配基因嵌入: {len(out)}/{len(target_genes)} (dim={emb_w.shape[1]})")
    return out


def load_esm2_gene_embeddings(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    genes = d["genes"]
    embs = d["embeddings"]
    return {str(g): embs[i].astype(np.float32) for i, g in enumerate(genes)}


def load_scfoundation_gene_embeddings(tsv_path: str, pt_path: str, target_genes: list) -> dict:
    df = pd.read_csv(tsv_path, sep="\t")
    g2idx = {str(r["gene_name"]): int(r["index"]) for _, r in df.iterrows()}
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    pos_w = ck["pos_emb.weight"].float().numpy()
    out = {}
    for g in target_genes:
        idx = g2idx.get(g) or g2idx.get(g.upper())
        if idx is not None and idx < pos_w.shape[0]:
            v = pos_w[idx].copy()
            out[g] = v / (np.linalg.norm(v) + 1e-6)
    print(f"[scFoundation] 成功匹配基因嵌入: {len(out)}/{len(target_genes)} (dim={pos_w.shape[1]})")
    return out


class MultiTeacherStudent(nn.Module):
    """支持任意维度多教师融合先验的轻量学生模型。"""
    def __init__(self, n_genes: int, n_tokens: int, prior_mat: np.ndarray, hidden_dim: int = 256):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior_mat, dtype=torch.float32))
        self.tok_emb = nn.Embedding(n_tokens + 1, 64, padding_idx=0)
        self.prior_proj = nn.Linear(prior_mat.shape[1], hidden_dim - 64)
        self.pert_enc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.gate_raw = nn.Parameter(torch.zeros(1, n_genes))

    def forward(self, x_ctrl_mean, token_ids, delta_lin):
        xz = (x_ctrl_mean - x_ctrl_mean.mean(dim=-1, keepdim=True)) / (x_ctrl_mean.std(dim=-1, keepdim=True) + 1e-6)
        z_ctx = self.ctx_enc(xz)
        mask = (token_ids == 0)
        valid = (~mask).unsqueeze(-1).float()
        h_tok = self.pert_enc(torch.cat([self.tok_emb(token_ids), self.prior_proj(self.prior[token_ids])], dim=-1))
        h_sum = (h_tok * valid).sum(dim=1)
        if token_ids.shape[1] >= 2:
            h_pair = (h_tok[:, 0, :] * h_tok[:, 1, :]) * (valid[:, 0, :] * valid[:, 1, :])
        else:
            h_pair = torch.zeros_like(h_sum)
        feat = self.fusion(torch.cat([z_ctx, h_sum, h_pair], dim=-1))
        gate = 1.0 + 0.25 * torch.tanh(self.gate_raw)
        return delta_lin * gate + self.delta_head(feat)


def eval_teacher_config(adata, ensembl_col, teacher_dicts: dict, teacher_name: str, dataset_label: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = adata[adata.obs["ood_split"] == "train"]
    va = adata[adata.obs["ood_split"] == "val"]
    ev = adata[adata.obs["ood_split"] == "test"]

    def get_cm(sub):
        m = (sub.obs["perturbation"].astype(str) == "control").values
        return _to_dense_array(sub.X[m]).mean(axis=0)

    mu_tr, mu_va, mu_ev = get_cm(tr), get_cm(va), get_cm(ev)
    tr_d = _compute_group_deltas(tr[tr.obs["perturbation"] != "control"], "perturbation", mu_tr)
    va_d = _compute_group_deltas(va[va.obs["perturbation"] != "control"], "perturbation", mu_va)
    ev_d = _compute_group_deltas(ev[ev.obs["perturbation"] != "control"], "perturbation", mu_ev)

    def parse_toks(ps):
        return [t.strip() for t in ps.split("_") if t.strip()]

    all_p = sorted(set(tr_d) | set(va_d) | set(ev_d))
    vocab = sorted({t for p in all_p for t in parse_toks(p)})
    t2id = {t: i + 1 for i, t in enumerate(vocab)}
    max_len = max(len(parse_toks(p)) for p in all_p)

    # 拼接多教师嵌入矩阵
    dims = [len(next(iter(d.values()))) for d in teacher_dicts.values()]
    p_mat = np.zeros((len(vocab) + 1, sum(dims)), dtype=np.float32)
    for t, tid in t2id.items():
        parts = []
        for d, dm in zip(teacher_dicts.values(), dims):
            parts.append(d.get(t, np.zeros(dm, dtype=np.float32)))
        p_mat[tid] = np.concatenate(parts, axis=0)

    tr_vocab = sorted({t for p in tr_d for t in parse_toks(p)})
    lin_idx = {t: i for i, t in enumerate(tr_vocab)}
    def enc_lin(plist):
        m = np.zeros((len(plist), len(tr_vocab)), dtype=np.float32)
        for ri, ps in enumerate(plist):
            for t in parse_toks(ps):
                if t in lin_idx:
                    m[ri, lin_idx[t]] = 1.0
        return m

    tr_keys = sorted(tr_d.keys())
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(enc_lin(tr_keys), np.vstack([tr_d[p] for p in tr_keys]))

    def make_t(d_dict, mu_c):
        ks = sorted(d_dict.keys())
        tids = np.zeros((len(ks), max_len), dtype=np.int64)
        for i, p in enumerate(ks):
            for j, t in enumerate(parse_toks(p)[:max_len]):
                tids[i, j] = t2id.get(t, 0)
        dl = ridge.predict(enc_lin(ks)).astype(np.float32)
        yt = np.vstack([d_dict[p] for p in ks]).astype(np.float32)
        xc = np.tile(mu_c, (len(ks), 1)).astype(np.float32)
        return (ks, torch.tensor(xc, device=device), torch.tensor(tids, device=device), torch.tensor(dl, device=device), torch.tensor(yt, device=device))

    tr_k, tr_xc, tr_tid, tr_dl, tr_y = make_t(tr_d, mu_tr)
    va_k, va_xc, va_tid, va_dl, va_y = make_t(va_d, mu_va)
    ev_k, ev_xc, ev_tid, ev_dl, ev_y = make_t(ev_d, mu_ev)

    torch.manual_seed(42)
    net = MultiTeacherStudent(adata.n_vars, len(vocab), p_mat, hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=2e-3)
    best_score = -1e9
    best_st = None
    for ep in range(1, 101):
        net.train()
        opt.zero_grad()
        pred = net(tr_xc, tr_tid, tr_dl)
        loss = F.smooth_l1_loss(pred, tr_y, beta=0.1) + 25.0 * F.mse_loss(pred, tr_y) + 0.8 * (1.0 - F.cosine_similarity(pred, tr_y, dim=-1)).mean()
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                vp = net(va_xc, va_tid, va_dl).cpu().numpy()
                dfv = evaluate_perturbation_predictions(va_d, {p: vp[i] for i, p in enumerate(va_k)}, teacher_name, (20, 50))
                sc_v = float(dfv["pearson_delta_all"].mean() + 0.5 * dfv["pearson_delta_top20"].mean() - 10.0 * dfv["mse_all"].mean())
                if sc_v > best_score:
                    best_score = sc_v
                    best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_st is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    net.eval()
    with torch.no_grad():
        ep_np = net(ev_xc, ev_tid, ev_dl).cpu().numpy()
    dfe = evaluate_perturbation_predictions(ev_d, {p: ep_np[i] for i, p in enumerate(ev_k)}, teacher_name, (20, 50))
    r_all = float(dfe["pearson_delta_all"].mean())
    r_top20 = float(dfe["pearson_delta_top20"].mean())
    mse_all = float(dfe["mse_all"].mean())
    print(f"  [{dataset_label}] {teacher_name}: Pearson All={r_all:.4f} | Top20={r_top20:.4f} | MSE={mse_all:.6f}")
    return {"dataset": dataset_label, "model": teacher_name, "pearson_delta_all": r_all, "pearson_delta_top20": r_top20, "mse_all": mse_all}


if __name__ == "__main__":
    esm2_dict = load_esm2_gene_embeddings(r"D:\data\AIVC\data\target_esm2_150m.npz")
    gf_dir = r"D:\data\AIVC\pretrained\geneformer\Geneformer-V2-316M"
    tok_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\token_dictionary_gc104M.pkl"
    gn_pkl = r"D:\data\AIVC\pretrained\geneformer\geneformer\gene_name_id_dict_gc104M.pkl"

    rows = []
    for ds_label, h5_path, ens_col in [
        ("Norman2019", r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad", "ensemble_id"),
        ("Shifrut2018", r"D:\data\AIVC\data\Shifrut2018_hvg2000_split.h5ad", "ensembl_id"),
    ]:
        print(f"\n=== 正在评测多教师集成蒸馏: {ds_label} ===")
        adata = sc.read_h5ad(h5_path)
        all_non_ctrl = sorted(set(adata.obs.loc[adata.obs["perturbation"] != "control", "perturbation"].astype(str)))
        vocab = sorted({t.strip() for p in all_non_ctrl for t in p.split("_") if t.strip()})
        var_ens = {g: str(adata.var[ens_col].iloc[i]) for i, g in enumerate(adata.var_names)} if ens_col in adata.var.columns else None

        gf_dict = extract_geneformer_gene_embeddings(gf_dir, tok_pkl, gn_pkl, vocab, var_ens)
        scgpt_dict = load_scgpt_gene_embeddings(r"D:\data\AIVC\pretrained\scgpt\vocab.json", r"D:\data\AIVC\pretrained\scgpt\model.safetensors", vocab)
        scf_dict = load_scfoundation_gene_embeddings(r"D:\data\AIVC\pretrained\scfoundation\OS_scRNA_gene_index.19264.tsv", r"D:\data\AIVC\pretrained\scfoundation\model.pt", vocab)

        rows.append(eval_teacher_config(adata, ens_col, {"scGPT": scgpt_dict}, "Teacher-scGPT(512d)", ds_label))
        rows.append(eval_teacher_config(adata, ens_col, {"scFoundation": scf_dict}, "Teacher-scFoundation(768d)", ds_label))
        rows.append(eval_teacher_config(adata, ens_col, {"ESM2": esm2_dict}, "Teacher-ESM2(640d)", ds_label))
        rows.append(eval_teacher_config(adata, ens_col, {"GF316M": gf_dict}, "Teacher-GF316M(1152d)", ds_label))
        rows.append(eval_teacher_config(adata, ens_col, {"GF316M": gf_dict, "scGPT": scgpt_dict, "scFoundation": scf_dict, "ESM2": esm2_dict}, "QuadTeacher-Ensemble(GF316M+scGPT+scFoundation+ESM2)", ds_label))

    res_df = pd.DataFrame(rows)
    out_csv = r"D:\data\AIVC\test_results\exp3_triteacher_distill_summary.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n=== 三教师集成蒸馏汇总 ===")
    print(res_df.to_string(index=False))
