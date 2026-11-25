import scanpy as sc
ad = sc.read_h5ad(r"D:\data\AIVC\data\MIXseq_hvg2000.h5ad", backed="r")
obs = ad.obs[["time", "perturbation", "dose_value", "cell_line", "channel", "hash_tag"]].copy()
ad.file.close()
sub = obs[obs["time"].astype(str) == "3, 6, 12, 24, 48"]
print("Pool '3, 6, 12, 24, 48' shape:", len(sub))
print("  perturbations:", sub["perturbation"].astype(str).value_counts().to_dict())
print("  cell_lines:", sub["cell_line"].astype(str).nunique())
print("  channels:", sub["channel"].astype(str).value_counts().head(10).to_dict())
print("  hash_tags sample:", sub["hash_tag"].astype(str).value_counts().head(10).to_dict())
