# -*- coding: utf-8 -*-
"""
boltz2_mace_mutant_resistance_test.py
AI 制药二次开发验证 1：患者驱动突变个性化共折叠 + MACE-OFF 量子力场复核
验证野生型 EGFR vs 临床耐药突变 EGFR T790M 面对：
  - 一代 EGFR 抑制剂 Gefitinib (临床上 T790M 突变会导致耐药失效)
  - 三代 EGFR 抑制剂 Osimertinib (临床上专门克服 T790M 耐药)
"""

import os
import json
import glob
import subprocess
import pandas as pd
import requests
from mace.calculators import mace_off

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mace_off_pocket_md_smoke import run_case

CACHE_DIR = r"D:\data\AIVC\pretrained\boltz_cache"
WORK_DIR = r"D:\data\AIVC\test_results\boltz2_mutant_test"
YAML_DIR = os.path.join(WORK_DIR, "inputs")
OUT_DIR = os.path.join(WORK_DIR, "outputs")
BOLTZ_EXE = r"D:\omics_tools\miniconda3\envs\boltz_md\Scripts\boltz.exe"
OUT_CSV = os.path.join(WORK_DIR, "egfr_wt_vs_t790m_boltz2_mace.csv")


def fetch_egfr_wt_and_t790m(start_1based: int = 696, end_1based: int = 1022) -> tuple[str, str]:
    resp = requests.get("https://rest.uniprot.org/uniprotkb/P00533.fasta", timeout=30)
    resp.raise_for_status()
    full_seq = "".join(ln.strip() for ln in resp.text.splitlines() if ln.strip() and not ln.startswith(">"))
    # UniProt P00533 canonical sequence contains 24-aa signal peptide: T790 in mature numbering is 790 in canonical? Wait -> check!
    idx_mature = 790 - 1
    idx_pro = 790 + 24 - 1
    if full_seq[idx_mature] == "T":
        mut_pos = idx_mature
    elif full_seq[idx_pro] == "T":
        mut_pos = idx_pro
    else:
        raise ValueError(f"Cannot locate Thr790 in P00533 (790={full_seq[idx_mature]}, 814={full_seq[idx_pro]})")
    print(f"[UniProt P00533] 精确验证第 {mut_pos+1} 位残基为 '{full_seq[mut_pos]}' -> 构建 T->M 耐药突变")
    mut_seq = full_seq[:mut_pos] + "M" + full_seq[mut_pos + 1 :]
    return full_seq[start_1based - 1 : end_1based], mut_seq[start_1based - 1 : end_1based]


def write_yaml(fpath: str, prot_seq: str, smiles: str):
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {prot_seq}\n"
            "      msa: empty\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{smiles}'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    wt_kd, t790m_kd = fetch_egfr_wt_and_t790m(696, 1022)

    drugs = {
        "Gefitinib_1stGen": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "Osimertinib_3rdGen": "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C)OC",
    }
    cases = [
        ("EGFR_WT__Gefitinib", wt_kd, drugs["Gefitinib_1stGen"]),
        ("EGFR_T790M__Gefitinib", t790m_kd, drugs["Gefitinib_1stGen"]),
        ("EGFR_WT__Osimertinib", wt_kd, drugs["Osimertinib_3rdGen"]),
        ("EGFR_T790M__Osimertinib", t790m_kd, drugs["Osimertinib_3rdGen"]),
    ]
    for cname, seq, smi in cases:
        write_yaml(os.path.join(YAML_DIR, f"{cname}.yaml"), seq, smi)

    cmd = [
        BOLTZ_EXE, "predict", YAML_DIR,
        "--out_dir", OUT_DIR, "--cache", CACHE_DIR,
        "--model", "boltz2", "--accelerator", "gpu",
        "--recycling_steps", "2", "--sampling_steps", "100", "--diffusion_samples", "1",
        "--sampling_steps_affinity", "100", "--diffusion_samples_affinity", "2",
        "--affinity_mw_correction", "--output_format", "pdb",
        "--num_workers", "0", "--no_kernels", "--override",
    ]
    existing_aff = glob.glob(os.path.join(OUT_DIR, "**", "affinity_*.json"), recursive=True)
    if len(existing_aff) < len(cases):
        print("[Mutant Test] 启动 Boltz-2 4组共折叠与亲和力推理...", flush=True)
        subprocess.run(cmd, check=True)
    else:
        print("[Mutant Test] 检测到 4 组 Boltz-2 结果已生成，直接进入 MACE-OFF23 量子力场复核...", flush=True)

    calc = mace_off(model="small", device="cuda", default_dtype="float32")
    rows = []
    for cname, _, smi in cases:
        aff_f = glob.glob(os.path.join(OUT_DIR, "**", f"affinity_{cname}.json"), recursive=True)
        conf_f = glob.glob(os.path.join(OUT_DIR, "**", f"confidence_{cname}_model_0.json"), recursive=True)
        pdb_f = glob.glob(os.path.join(OUT_DIR, "**", f"{cname}_model_0.pdb"), recursive=True)
        row = {"complex": cname}
        if aff_f:
            with open(aff_f[0], "r", encoding="utf-8") as f:
                aff = json.load(f)
            row["affinity_pred_value"] = round(float(aff.get("affinity_pred_value", 0.0)), 4)
            row["affinity_probability_binary"] = round(float(aff.get("affinity_probability_binary", 0.0)), 4)
        if conf_f:
            with open(conf_f[0], "r", encoding="utf-8") as f:
                conf = json.load(f)
            row["ligand_iptm"] = round(float(conf.get("ligand_iptm", 0.0)), 4)
        if pdb_f:
            md_res = run_case(cname, pdb_f[0], calc)
            row["mace_bind_energy_kcal"] = md_res["mace_bind_energy_kcal"]
            row["md_mean_rmsd_A"] = md_res["md_mean_rmsd_A"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== EGFR WT vs T790M 耐药突变个性化筛选对比 ({OUT_CSV}) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
