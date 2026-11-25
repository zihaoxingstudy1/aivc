import torch
ck = torch.load(r"D:\data\AIVC\pretrained\aivc_5omics_pathway_gnn.pt", map_location="cpu", weights_only=False)
print("type:", type(ck))
if isinstance(ck, dict):
    print("keys:", list(ck.keys()))
    for k in ["top_rna", "rppa_abs", "top_prot", "top_phos", "tf_list"]:
        if k in ck:
            print(f"  {k}: len={len(ck[k])}, head={ck[k][:5]}")
