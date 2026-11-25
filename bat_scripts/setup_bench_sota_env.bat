@echo off
echo === Setup SOTA Open-Source Benchmark env bench_sota (GEARS + scGPT + scVI) ===

set "CONDA_EXE=D:\omics_tools\miniconda3\Scripts\conda.exe"
set "ENV_PY=D:\omics_tools\miniconda3\envs\bench_sota\python.exe"

echo [1/3] Ensure conda env bench_sota (Python 3.11) exists...
if not exist "%ENV_PY%" call "%CONDA_EXE%" create -n bench_sota python=3.11 pip -y

echo [2/3] Verify packages without old cpa-tools 0.0.0 stub...
call "%ENV_PY%" -m pip uninstall -y cpa-tools
call "%ENV_PY%" -m pip install "setuptools<81" "jax==0.4.35" "jaxlib==0.4.35" "orbax-checkpoint==0.5.20" "flax<0.10" "scvi-tools==1.1.6" cell-gears scanpy anndata scikit-learn scipy pandas torchtext rdkit ipython -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
call "%ENV_PY%" -m pip install --no-deps scgpt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo [3/3] Verify GEARS, scGPT, scVI, CUDA...
call "%ENV_PY%" -c "import torch, gears, scvi, scgpt; print('CUDA:', torch.cuda.is_available(), '| GEARS OK | scVI:', scvi.__version__, '| scGPT OK')"
echo === bench_sota setup finished ===
