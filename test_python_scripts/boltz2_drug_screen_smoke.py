# -*- coding: utf-8 -*-
"""
boltz2_drug_screen_smoke.py
在独立 GPU 环境 boltz_md 中运行一例真实的 Boltz-2 药物筛选冒烟对比：
  靶点：人类 EGFR 激酶结构域 (UniProt P00533, 696-1022 aa)
  对比药物：Osimertinib (三代 EGFR 靶向药) vs Aspirin (阴性小分子对照)
  输出：3D 复合物 PDB + 结合概率 (affinity_probability_binary) + 亲和力 (affinity_pred_value)
"""

import os
import json
import glob
import tarfile
import subprocess
import requests
import pandas as pd

CACHE_DIR = r"D:\data\AIVC\pretrained\boltz_cache"
WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def safe_extract_mols_tar(cache_dir: str):
    """安全解压 Boltz-2 的 mols.tar，跳过 Windows 保留设备名 (PRN/CON/AUX/NUL 等)。"""
    tar_path = os.path.join(cache_dir, "mols.tar")
    mols_dir = os.path.join(cache_dir, "mols")
    if not os.path.exists(tar_path):
        return
    if (
        os.path.exists(os.path.join(mols_dir, "CYS.pkl"))
        and os.path.exists(os.path.join(mols_dir, "TRP.pkl"))
        and len(os.listdir(mols_dir)) > 30000
    ):
        return
    print("[Boltz-2 Smoke] 安全解压 mols.tar (自动跳过 Windows 保留设备名如 PRN.pkl)...")
    os.makedirs(mols_dir, exist_ok=True)
    skipped = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            base = os.path.splitext(os.path.basename(member.name))[0].upper()
            if base in WIN_RESERVED:
                skipped += 1
                continue
            tar.extract(member, path=cache_dir)
    print(f"  -> 解压完成，跳过保留名文件数={skipped}")
WORK_DIR = r"D:\data\AIVC\test_results\boltz2_smoke"
YAML_DIR = os.path.join(WORK_DIR, "inputs")
OUT_DIR = os.path.join(WORK_DIR, "outputs")
BOLTZ_EXE = r"D:\omics_tools\miniconda3\envs\boltz_md\Scripts\boltz.exe"


def fetch_uniprot_seq(uniprot_id: str, start_1based: int, end_1based: int) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip() and not ln.startswith(">")]
    full_seq = "".join(lines)
    return full_seq[start_1based - 1 : end_1based]


def write_boltz_yaml(fpath: str, prot_seq: str, ligand_smiles: str):
    content = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {prot_seq}\n"
        "      msa: empty\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{ligand_smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(YAML_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    safe_extract_mols_tar(CACHE_DIR)

    print("[Boltz-2 Smoke] 获取人类 EGFR (P00533, 696-1022) 激酶域序列...")
    egfr_kd = fetch_uniprot_seq("P00533", 696, 1022)
    print(f"  -> EGFR Kinase Domain 长度 = {len(egfr_kd)} aa")

    candidates = [
        ("EGFR_Osimertinib", "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C)OC"),
        ("EGFR_Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]
    for cname, smi in candidates:
        write_boltz_yaml(os.path.join(YAML_DIR, f"{cname}.yaml"), egfr_kd, smi)

    cmd = [
        BOLTZ_EXE,
        "predict",
        YAML_DIR,
        "--out_dir", OUT_DIR,
        "--cache", CACHE_DIR,
        "--model", "boltz2",
        "--accelerator", "gpu",
        "--recycling_steps", "2",
        "--sampling_steps", "100",
        "--diffusion_samples", "1",
        "--sampling_steps_affinity", "100",
        "--diffusion_samples_affinity", "2",
        "--affinity_mw_correction",
        "--output_format", "pdb",
        "--num_workers", "0",
        "--no_kernels",
        "--override",
    ]
    print("[Boltz-2 Smoke] 启动 GPU 推理:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    rows = []
    for cname, smi in candidates:
        aff_files = glob.glob(os.path.join(OUT_DIR, "**", f"affinity_{cname}.json"), recursive=True)
        conf_files = glob.glob(os.path.join(OUT_DIR, "**", f"confidence_{cname}_model_0.json"), recursive=True)
        pdb_files = glob.glob(os.path.join(OUT_DIR, "**", f"{cname}_model_0.pdb"), recursive=True)
        row = {"complex": cname, "smiles": smi}
        if aff_files:
            with open(aff_files[0], "r", encoding="utf-8") as f:
                aff = json.load(f)
            row["affinity_pred_value"] = aff.get("affinity_pred_value")
            row["affinity_probability_binary"] = aff.get("affinity_probability_binary")
        if conf_files:
            with open(conf_files[0], "r", encoding="utf-8") as f:
                conf = json.load(f)
            row["confidence_score"] = conf.get("confidence_score")
            row["ligand_iptm"] = conf.get("ligand_iptm")
        row["pdb_path"] = pdb_files[0] if pdb_files else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(WORK_DIR, "boltz2_smoke_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n=== Boltz-2 药物筛选冒烟结果 ({out_csv}) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
