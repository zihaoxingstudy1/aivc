# -*- coding: utf-8 -*-
"""
从 PubChem 批量查询 sci-Plex3 188 种小分子药物的真实 Canonical SMILES 并保存至本地 CSV
"""

import os
import time
import json
import urllib.request
import urllib.parse
import pandas as pd
import anndata as ad


def query_pubchem_smiles(drug_name: str) -> str:
    candidates = [drug_name.strip()]
    if "(" in drug_name and ")" in drug_name:
        main_part = drug_name.split("(")[0].strip()
        paren_part = drug_name.split("(")[1].split(")")[0].strip()
        candidates.extend([main_part, paren_part])

    for name in candidates:
        if not name:
            continue
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
        try:
            req = urllib.request.urlopen(url, timeout=8)
            data = json.loads(req.read().decode("utf-8"))
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("CanonicalSMILES") or props[0].get("ConnectivitySMILES") or ""
        except Exception:
            continue
    return ""


if __name__ == "__main__":
    a = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000_split.h5ad", backed="r")
    meta = a.obs[["perturbation", "target", "pathway_level_1"]].drop_duplicates()
    meta = meta[meta["perturbation"].astype(str) != "control"].copy()
    drugs = sorted(meta["perturbation"].astype(str).unique())
    print(f"[SMILES] 待查询药物总数: {len(drugs)}")

    rows = []
    hit = 0
    for idx, d in enumerate(drugs):
        smi = query_pubchem_smiles(d)
        if smi:
            hit += 1
        sub_m = meta[meta["perturbation"].astype(str) == d].iloc[0]
        rows.append({
            "perturbation": d,
            "smiles": smi,
            "target": str(sub_m["target"]),
            "pathway_level_1": str(sub_m["pathway_level_1"]),
        })
        if (idx + 1) % 30 == 0:
            print(f"[SMILES] 进度 {idx+1}/{len(drugs)}, 已命中 {hit}")
        time.sleep(0.15)

    out_csv = r"D:\data\AIVC\data\sciplex3_drug_smiles.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[SMILES] 查询完成: 命中 {hit}/{len(drugs)} 种药物 SMILES, 已保存至 {out_csv}")
