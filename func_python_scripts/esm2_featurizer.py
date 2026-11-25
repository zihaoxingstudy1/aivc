# -*- coding: utf-8 -*-
"""
使用 UniProt REST API + facebook/esm2_t30_150M_UR50D 提取人类靶点基因的 640 维蛋白语言模型嵌入
"""

import os
import json
import urllib.request
import urllib.parse
import numpy as np
import scanpy as sc
import torch
from transformers import AutoTokenizer, AutoModel


def fetch_protein_seqs_from_uniprot(gene_names: list) -> dict:
    """通过 UniProtKB REST API 批量获取人类 Reviewed (Swiss-Prot) 蛋白序列。"""
    gene_to_seq = {}
    batch_size = 25
    for i in range(0, len(gene_names), batch_size):
        chunk = gene_names[i : i + batch_size]
        or_expr = " OR ".join(f"gene_exact:{g}" for g in chunk)
        q = f"({or_expr}) AND organism_id:9606 AND reviewed:true"
        url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            + urllib.parse.urlencode({"query": q, "fields": "gene_primary,sequence", "format": "json", "size": 100})
        )
        try:
            resp = urllib.request.urlopen(url, timeout=15)
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results", []):
                seq = item.get("sequence", {}).get("value", "")
                if not seq:
                    continue
                for g_obj in item.get("genes", []):
                    gname = g_obj.get("geneName", {}).get("value", "")
                    if gname and (gname not in gene_to_seq or len(seq) > len(gene_to_seq[gname])):
                        gene_to_seq[gname] = seq
        except Exception as e:
            print(f"[ESM2] UniProt batch {i} 警告: {str(e)[:100]}")
    return gene_to_seq


@torch.no_grad()
def extract_esm2_gene_embeddings(
    target_genes: list,
    esm2_dir: str,
    save_npz_path: str,
    max_aa_len: int = 1022,
):
    print(f"[ESM2] 正在从 UniProt 获取 {len(target_genes)} 个靶点基因的人类蛋白序列...")
    gene_to_seq = fetch_protein_seqs_from_uniprot(target_genes)
    print(f"[ESM2] 成功获取蛋白序列: {len(gene_to_seq)}/{len(target_genes)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(esm2_dir)
    model = AutoModel.from_pretrained(esm2_dir, torch_dtype=torch.float16).to(device)
    model.eval()

    genes = sorted(gene_to_seq.keys())
    seqs = [gene_to_seq[g][:max_aa_len] for g in genes]

    embs_list = []
    for i in range(0, len(seqs), 16):
        b_seqs = seqs[i : i + 16]
        enc = tokenizer(b_seqs, padding=True, truncation=True, max_length=max_aa_len + 2, return_tensors="pt").to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state.float() * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        embs_list.append(emb.cpu().numpy().astype(np.float32))

    emb_mat = np.vstack(embs_list) if embs_list else np.zeros((0, 640), dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(save_npz_path)), exist_ok=True)
    np.savez(save_npz_path, genes=np.array(genes), embeddings=emb_mat)
    print(f"[ESM2] 已保存 {len(genes)} 个靶点基因的 640 维蛋白嵌入至: {save_npz_path}")


if __name__ == "__main__":
    target_genes = set()
    for p_h5 in [
        r"D:\data\AIVC\data\Norman2019_hvg2000_split.h5ad",
        r"D:\data\AIVC\data\Tian2021_hvg2000_split.h5ad",
        r"D:\data\AIVC\data\Shifrut2018_hvg2000_split.h5ad",
    ]:
        a = sc.read_h5ad(p_h5, backed="r")
        for p in a.obs["perturbation"].astype(str).unique():
            if p == "control":
                continue
            for tok in p.split("_"):
                t = tok.strip()
                if t:
                    target_genes.add(t)

    extract_esm2_gene_embeddings(
        target_genes=sorted(target_genes),
        esm2_dir=r"D:\data\AIVC\pretrained\esm2_150m",
        save_npz_path=r"D:\data\AIVC\data\target_esm2_150m.npz",
    )
