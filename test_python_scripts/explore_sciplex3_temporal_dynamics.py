# -*- coding: utf-8 -*-
"""
explore_sciplex3_temporal_dynamics.py
在动任何时序动力学模型代码之前，先用真实数据探查 SrivatsanTrapnell2020_sciplex3.h5ad 中
47 种药物 x 4 剂量在 0h -> 24h -> 72h 的真实表达演化、增殖/凋亡/代偿通路时序规律。
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr

H5AD_PATH = r"D:\data\AIVC\data\SrivatsanTrapnell2020_sciplex3.h5ad"
HVG_REF = r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad"
PRIORS_MEM = r"D:\data\AIVC\data\priors\gene_pathway_membership.parquet"
OUT_CSV = r"D:\data\AIVC\test_results\sciplex3_24h_vs_72h_real_dynamics.csv"


def main():
    ref = sc.read_h5ad(HVG_REF, backed="r")
    hvg_genes = list(ref.var_names)
    ref.file.close()

    print(f"[Explore] 读取 Sciplex3 A549 (对齐 {len(hvg_genes)} HVG) ...")
    ad = sc.read_h5ad(H5AD_PATH, backed="r")
    obs = ad.obs[["cell_line", "time", "perturbation", "dose_value"]].copy()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)
    mask_a549 = (obs["cell_line"] == "A549") & (obs["time"].isin(["24.0", "72.0"]))
    sub = ad[mask_a549.values, hvg_genes].to_memory()
    ad.file.close()
    print(f"  -> A549 细胞数={sub.n_obs}, 基因数={sub.n_vars}")

    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    sub_obs = sub.obs.copy()
    for c in ["time", "perturbation", "dose_value"]:
        sub_obs[c] = sub_obs[c].astype(str)

    # 分别计算 24h 与 72h 的 control 均值基线
    ctrl_24 = np.asarray(sub[(sub_obs["perturbation"] == "control") & (sub_obs["time"] == "24.0")].X.mean(axis=0)).ravel()
    ctrl_72 = np.asarray(sub[(sub_obs["perturbation"] == "control") & (sub_obs["time"] == "72.0")].X.mean(axis=0)).ravel()
    print(f"  -> Control 细胞数: 24h={( (sub_obs['perturbation']=='control') & (sub_obs['time']=='24.0') ).sum()}, 72h={( (sub_obs['perturbation']=='control') & (sub_obs['time']=='72.0') ).sum()}")

    # 载入 Hallmark 关键通路用于观察真实通路时序动态
    mem = pd.read_parquet(PRIORS_MEM)
    hm = mem[mem["source_db"] == "MSigDB_Hallmark_2020"]
    g2i = {g.upper(): i for i, g in enumerate(hvg_genes)}
    pw_idx = {}
    for pw, grp in hm.groupby("pathway_name"):
        idxs = [g2i[g.upper()] for g in grp["gene_symbol"] if g.upper() in g2i]
        if len(idxs) >= 5:
            pw_idx[pw] = idxs

    # 统计同时有 24h 与 72h 且各 >= 20 细胞的 (drug, dose) 轨迹
    rows = []
    tr_obs = sub_obs[sub_obs["perturbation"] != "control"]
    grouped = tr_obs.groupby(["perturbation", "dose_value", "time"]).indices
    pairs = sorted({(k[0], k[1]) for k in grouped.keys()})

    for drug, dose in pairs:
        i24 = grouped.get((drug, dose, "24.0"))
        i72 = grouped.get((drug, dose, "72.0"))
        if i24 is None or i72 is None or len(i24) < 20 or len(i72) < 20:
            continue
        d24 = np.asarray(sub.X[i24].mean(axis=0)).ravel() - ctrl_24
        d72 = np.asarray(sub.X[i72].mean(axis=0)).ravel() - ctrl_72
        n24 = float(np.linalg.norm(d24))
        n72 = float(np.linalg.norm(d72))
        r_24_72 = float(pearsonr(d24, d72)[0]) if n24 > 1e-6 and n72 > 1e-6 else 0.0
        amp_ratio = float(n72 / max(n24, 1e-6))

        row = {
            "drug": drug,
            "dose_nM": dose,
            "n_24h": len(i24),
            "n_72h": len(i72),
            "norm_24h": round(n24, 4),
            "norm_72h": round(n72, 4),
            "amp_ratio_72_vs_24": round(amp_ratio, 4),
            "pearson_24_vs_72": round(r_24_72, 4),
        }
        for pw_key in ["HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT", "HALLMARK_APOPTOSIS", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"]:
            if pw_key in pw_idx:
                idxs = pw_idx[pw_key]
                row[f"{pw_key}_24h"] = round(float(d24[idxs].mean()), 4)
                row[f"{pw_key}_72h"] = round(float(d72[idxs].mean()), 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("norm_72h", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== Sciplex3 A549 真实 24h -> 72h 轨迹配对统计 (有效配对={len(df)}, 覆盖药物={df['drug'].nunique()}) ===")
    print(f"24h vs 72h 平均相关系数 r = {df['pearson_24_vs_72'].mean():.4f} | 平均幅值比 (72h/24h) = {df['amp_ratio_72_vs_24'].mean():.4f}")
    print("\n--- 72h 效应最显著的 Top 10 轨迹 (看 24h->72h 深化 vs 回弹) ---")
    show_cols = ["drug", "dose_nM", "norm_24h", "norm_72h", "amp_ratio_72_vs_24", "pearson_24_vs_72"]
    print(df[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
