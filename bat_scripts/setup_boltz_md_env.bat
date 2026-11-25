@echo off
echo === Setup AI Drug Discovery GPU env boltz_md (Boltz-2 + OpenMM + MACE-OFF) ===

set "CONDA_EXE=D:\omics_tools\miniconda3\Scripts\conda.exe"
set "ENV_PY=D:\omics_tools\miniconda3\envs\boltz_md\python.exe"

echo [1/4] Configure Aliyun pip mirror...
call "%CONDA_EXE%" run -n base python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
call "%CONDA_EXE%" run -n base python -m pip config set global.trusted-host mirrors.aliyun.com

echo [2/4] Ensure boltz_md env has OpenMM + PDBFixer + pip...
if not exist "%ENV_PY%" call "%CONDA_EXE%" create -n boltz_md python=3.11 -y
call "%CONDA_EXE%" install -n boltz_md -c conda-forge openmm pdbfixer pip -y

echo [3/4] Install PyTorch CUDA + Boltz-2 + MACE-OFF + RDKit via pip (no pip self-upgrade on Windows)...
call "%ENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
call "%ENV_PY%" -m pip install boltz mace-torch rdkit -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo [4/4] Verify CUDA, OpenMM, Boltz-2, MACE-OFF, RDKit...
call "%ENV_PY%" -c "import torch, openmm, boltz, mace, rdkit; print('CUDA:', torch.cuda.is_available(), '| OpenMM:', openmm.__version__, '| Boltz:', boltz.__file__)"
echo === Setup finished ===
