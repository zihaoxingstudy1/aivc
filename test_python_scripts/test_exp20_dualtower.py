import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import torch, torch.nn as nn
from scipy.stats import pearsonr
import sys
sys.path.insert(0, r"D:\data\AIVC\func_python_scripts")
from clinical_cohort_cleaner import load_ensembl_to_sym_from_gtex

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

s3 = ad.read_h5ad(r"D:\data\AIVC\data\Sciplex3_24h_hvg2000.h5ad")
sciplex_genes = [str(g).upper() for g in s3.var_names]
X_s3 = s3.X.toarray().astype(np.float32) if sp.issparse(s3.X) else np.asarray(s3.X, dtype=np.float32)
obs = s3.obs.copy()
obs["pert_str"] = obs["perturbation"].astype(str)

m_npz = np.load(r"D:\data\AIVC\data\sciplex3_mol_features.npz", allow_pickle=True)
t_npz = np.load(r"D:\data\AIVC\data\sciplex3_target_esm2_kg.npz", allow_pickle=True)
sm_df = pd.read_csv(r"D:\data\AIVC\data\sciplex3_drug_smiles.csv")
drug_to_targ = dict(zip(sm_df["perturbation"].astype(str), sm_df["target"].astype(str)))
targ_to_emb = {str(t): t_npz["target_embs"][i].astype(np.float32) for i, t in enumerate(t_npz["targets"])}
dim_esm = t_npz["target_embs"].shape[1]

d_morgan = {str(d): np.concatenate([m_npz["morgan"][i], m_npz["chemberta"][i]]).astype(np.float32) for i, d in enumerate(m_npz["drugs"])}
d_esm = {
    str(d): np.concatenate([
        targ_to_emb.get(drug_to_targ.get(str(d), ""), np.zeros(dim_esm, dtype=np.float32)),
        t_npz["pathway_onehot"][i]
    ]).astype(np.float32)
    for i, d in enumerate(t_npz["drugs"])
}

ctrl_mask = obs["pert_str"].str.lower().isin(["vehicle", "dmso", "control"])
ctrl_by_cl = {str(cl): X_s3[(ctrl_mask & (obs["cell_line"] == cl)).values].mean(axis=0) for cl in obs["cell_line"].unique()}

rows = []
for (cl, pert, dose), idxs in obs[~ctrl_mask].groupby(["cell_line", "pert_str", "dose_value"], observed=True).indices.items():
    if len(idxs) < 10 or pert not in d_morgan or pert not in d_esm:
        continue
    xc = ctrl_by_cl[str(cl)]
    delta = X_s3[idxs].mean(axis=0) - xc
    log_d = float(np.log10(max(float(dose), 1.0)) / 4.0)
    rows.append((pert, float(dose), xc, d_morgan[pert], d_esm[pert], np.array([log_d], dtype=np.float32), delta))

ens_to_sym = load_ensembl_to_sym_from_gtex(r"D:\data\AIVC\data\GTEx_v8_gene_median_tpm.gct.gz")
df307 = pd.read_csv(r"D:\data\AIVC\data\wetlab_bulk_validation\GSE307106\GSE307106_raw_counts.txt.gz", sep="\t", index_col=0)
df307["__sym__"] = [ens_to_sym.get(str(i).split(".")[0], "") for i in df307.index]
df307 = df307[df307["__sym__"] != ""].groupby("__sym__", sort=False).mean()
vals = df307.values
mat307 = pd.DataFrame(np.log2(vals / np.maximum(vals.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0).T, index=df307.columns, columns=df307.index)
shared = [(j, g) for j, g in enumerate(sciplex_genes) if g in mat307.columns]
idx_s = [s[0] for s in shared]
cols_b = [s[1] for s in shared]
real_d = mat307.loc[["Panobinostat_20h_1", "Panobinostat_20h_2"], cols_b].mean(axis=0).values - mat307.loc[["DMSO_1", "DMSO_2"], cols_b].mean(axis=0).values
top50 = np.argsort(np.abs(real_d))[-50:]

class DualTowerPert(nn.Module):
    def __init__(self, n_genes, n_mol, n_kg):
        super().__init__()
        self.c_enc = nn.Sequential(nn.Linear(n_genes, 128), nn.LayerNorm(128), nn.GELU())
        self.m_enc = nn.Sequential(nn.Linear(n_mol, 128), nn.LayerNorm(128), nn.GELU())
        self.k_enc = nn.Sequential(nn.Linear(n_kg, 128), nn.LayerNorm(128), nn.GELU())
        self.d_enc = nn.Sequential(nn.Linear(1, 32), nn.GELU())
        self.dec = nn.Sequential(
            nn.Linear(128 * 3 + 32, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, n_genes)
        )
    def forward(self, xc, xm, xk, xd):
        return self.dec(torch.cat([self.c_enc(xc), self.m_enc(xm), self.k_enc(xk), self.d_enc(xd)], dim=-1))

# Ground truth Panobinostat 100nM delta in Sciplex3
pano_rows = [r[6] for r in rows if "Panobinostat" in r[0] and r[1] == 100.0]
gt_pano = np.mean(pano_rows, axis=0)

for holdout in [False, True]:
    tr = [r for r in rows if not (holdout and "Panobinostat" in r[0])]
    Xc = torch.tensor(np.stack([r[2] for r in tr]), device=DEVICE)
    Xm = torch.tensor(np.stack([r[3] for r in tr]), device=DEVICE)
    Xk = torch.tensor(np.stack([r[4] for r in tr]), device=DEVICE)
    Xd = torch.tensor(np.stack([r[5] for r in tr]), device=DEVICE)
    Yd = torch.tensor(np.stack([r[6] for r in tr]), device=DEVICE)

    net = DualTowerPert(Xc.shape[1], Xm.shape[1], Xk.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    cos = nn.CosineSimilarity(dim=-1)
    net.train()
    for _ in range(250):
        perm = torch.randperm(len(Xc), device=DEVICE)
        for i in range(0, len(Xc), 256):
            b = perm[i:i+256]
            opt.zero_grad()
            pred = net(Xc[b], Xm[b], Xk[b], Xd[b])
            loss = (pred - Yd[b]).pow(2).mean() + 0.5 * (1.0 - cos(pred, Yd[b])).mean()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p_name = "Panobinostat (LBH589)"
        xc_m = torch.tensor(np.mean(list(ctrl_by_cl.values()), axis=0), device=DEVICE).unsqueeze(0)
        xm_m = torch.tensor(d_morgan[p_name], device=DEVICE).unsqueeze(0)
        xk_m = torch.tensor(d_esm[p_name], device=DEVICE).unsqueeze(0)
        xd_m = torch.tensor([[np.log10(100.0)/4.0]], dtype=torch.float32, device=DEVICE)
        pred_full = net(xc_m, xm_m, xk_m, xd_m).squeeze(0).cpu().numpy()
    r_sc, _ = pearsonr(pred_full, gt_pano)
    pred_d = pred_full[idx_s]
    r_all, _ = pearsonr(pred_d, real_d)
    r_t50, _ = pearsonr(pred_d[top50], real_d[top50])
    dir_t50 = np.mean(np.sign(pred_d[top50]) == np.sign(real_d[top50]))
    print(f"DualTower (Holdout={holdout}): vs Sciplex3 GT r={r_sc:.4f} | vs GSE307106 Bulk All r={r_all:.4f} | Top50 r={r_t50:.4f}, DirAcc={dir_t50*100:.1f}%")
