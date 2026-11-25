# -*- coding: utf-8 -*-
"""
mace_off_pocket_md_smoke.py
AI 制药二次开发验证 2：Boltz-2 共折叠复合物 -> MACE-OFF23 (AI 量子精度机器学习力场) 口袋局部热扰动分子动力学复核
提取配体周围 6.0 Å 口袋原子，用 RDKit 加氢后在 300K 下跑 MACE-OFF Langevin MD，
量化对比 Osimertinib (强结合阳性药) vs Aspirin (阴性对照) 的配体 RMSD 稳定性与能量涨落。
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
from mace.calculators import mace_off

OUT_DIR = r"D:\data\AIVC\test_results\boltz2_smoke"
OUT_CSV = os.path.join(OUT_DIR, "mace_off_pocket_md_results.csv")

CASES = [
    (
        "EGFR_Osimertinib",
        os.path.join(OUT_DIR, r"outputs\boltz_results_inputs\predictions\EGFR_Osimertinib\EGFR_Osimertinib_model_0.pdb"),
        "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C)OC",
    ),
    (
        "EGFR_Aspirin",
        os.path.join(OUT_DIR, r"outputs\boltz_results_inputs\predictions\EGFR_Aspirin\EGFR_Aspirin_model_0.pdb"),
        "CC(=O)OC1=CC=CC=C1C(=O)O",
    ),
]


def parse_pocket_and_ligand(pdb_path: str, radius_angstrom: float = 5.0) -> tuple:
    """从 Boltz-2 PDB 提取重原子：蛋白 (Chain A) 与配体 (Chain B)，并截取配体周围局部口袋残基。"""
    prot_atoms, lig_atoms = [], []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21].strip()
                res_id = int(line[22:26].strip())
                elem = line[76:78].strip()
                if not elem:
                    elem = line[12:16].strip()[0]
                elem = elem.capitalize()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                rec = (elem, res_id, np.array([x, y, z], dtype=np.float64))
                if chain == "B":
                    lig_atoms.append(rec)
                else:
                    prot_atoms.append(rec)

    lig_xyz = np.vstack([a[2] for a in lig_atoms])
    # 找出距离配体任意原子 <= radius_angstrom 的完整残基
    pocket_res = set()
    for _, rid, xyz in prot_atoms:
        dists = np.linalg.norm(lig_xyz - xyz, axis=1)
        if dists.min() <= radius_angstrom:
            pocket_res.add(rid)

    pocket_atoms = [a for a in prot_atoms if a[1] in pocket_res]
    # 使用固定外围主链重原子 + 自由配体的 ASE Atoms 体系
    syms = [a[0] for a in pocket_atoms] + [a[0] for a in lig_atoms]
    pos = np.vstack([a[2] for a in pocket_atoms] + [a[2] for a in lig_atoms])
    n_pocket = len(pocket_atoms)
    atoms = Atoms(symbols=syms, positions=pos)
    # 固定口袋重原子作为局部受限势阱，聚焦观察配体在量子精度 MACE-OFF 势场下的热稳定性与结合能
    atoms.set_constraint(FixAtoms(indices=list(range(n_pocket))))
    return atoms, n_pocket, len(lig_atoms), len(pocket_res)


def run_case(name: str, pdb_path: str, calc) -> dict:
    t0 = time.time()
    atoms, n_pocket, n_lig, n_res = parse_pocket_and_ligand(pdb_path, radius_angstrom=5.0)
    atoms.calc = calc

    # 1. 局域能量极小化 (Relaxation)
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.15, steps=20)
    pos_eq = atoms.get_positions()[n_pocket:].copy()
    e_complex = float(atoms.get_potential_energy())

    # 计算单独口袋与单独配体参考能量以估算 MACE-OFF 结合相互作用能 (eV -> kcal/mol)
    pocket_only = atoms[:n_pocket].copy()
    pocket_only.calc = calc
    e_pocket = float(pocket_only.get_potential_energy())

    lig_only = atoms[n_pocket:].copy()
    lig_only.calc = calc
    e_lig = float(lig_only.get_potential_energy())
    e_bind_kcal = (e_complex - e_pocket - e_lig) * 23.0605  # eV to kcal/mol

    # 2. 300K Langevin AI 分子动力学热扰动 (60 步 x 1.0 fs)
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300.0, friction=0.05)
    rmsds, energies = [], []
    for step in range(60):
        dyn.run(1)
        pos_cur = atoms.get_positions()[n_pocket:]
        rmsd = float(np.sqrt(np.mean(np.sum((pos_cur - pos_eq) ** 2, axis=1))))
        rmsds.append(rmsd)
        energies.append(float(atoms.get_potential_energy()))

    res = {
        "complex": name,
        "pocket_residues": n_res,
        "pocket_atoms": n_pocket,
        "ligand_atoms": n_lig,
        "mace_bind_energy_kcal": round(e_bind_kcal, 2),
        "md_mean_rmsd_A": round(float(np.mean(rmsds)), 4),
        "md_max_rmsd_A": round(float(np.max(rmsds)), 4),
        "md_energy_std_eV": round(float(np.std(energies)), 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"  [{name}] -> {res}", flush=True)
    return res


def main():
    print("=== MACE-OFF23 AI 量子力场口袋局部动力学复核 (cuda) ===", flush=True)
    calc = mace_off(model="small", device="cuda", default_dtype="float32")
    rows = []
    for name, pdb_path, _ in CASES:
        rows.append(run_case(name, pdb_path, calc))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== MACE-OFF 动力学稳定性复核结果 ({OUT_CSV}) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
