import torch
import scvi
import scgpt
import gears
print("CUDA:", torch.cuda.is_available(), "| torch:", torch.__version__, "| scvi:", scvi.__version__)
try:
    import torch_geometric
    print("torch_geometric:", torch_geometric.__version__)
except Exception as e:
    print("torch_geometric missing:", e)
try:
    import safetensors.torch
    print("safetensors OK")
except Exception as e:
    print("safetensors missing:", e)
