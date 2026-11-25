import os
import numpy as np
import pandas as pd
import scipy.sparse as sp


def export_post_perturbation_h5ad(
    adata_pre,
    pred_delta: np.ndarray,
    pred_genes: list,
    pert_label: str,
    out_h5ad_path: str,
    out_deg_csv_path: str = None,
) -> tuple:
    """
    Apply predicted perturbation delta to a baseline AnnData (.h5ad) and export a standard
    Scanpy/Seurat-compatible virtual post-perturbation AnnData object.

    Parameters
    ----------
    adata_pre : anndata.AnnData
        Baseline (pre-perturbation / vehicle) single-cell AnnData object.
    pred_delta : np.ndarray
        Predicted expression change (1D shape (len(pred_genes),) or 2D shape (n_cells, len(pred_genes))).
    pred_genes : list
        Gene symbols corresponding to pred_delta columns.
    pert_label : str
        Name of the applied drug / combination / gene knockout perturbation.
    out_h5ad_path : str
        Output path for the generated virtual post-perturbation .h5ad file.
    out_deg_csv_path : str, optional
        Optional output CSV path for ranked predicted DEGs (compatible with bio_pathway_viz.py).

    Returns
    -------
    tuple : (adata_post, deg_df)
    """
    adata_post = adata_pre.copy()
    if sp.issparse(adata_post.X):
        x_pre = adata_post.X.toarray().astype(np.float32)
    else:
        x_pre = np.asarray(adata_post.X, dtype=np.float32)

    n_cells, n_total_genes = x_pre.shape
    delta_arr = np.asarray(pred_delta, dtype=np.float32)
    if delta_arr.ndim == 1:
        delta_arr = np.broadcast_to(delta_arr[None, :], (n_cells, len(pred_genes))).copy()

    var_upper = {str(g).strip().upper(): idx for idx, g in enumerate(adata_post.var_names)}
    full_delta = np.zeros((n_cells, n_total_genes), dtype=np.float32)
    is_pred_gene = np.zeros(n_total_genes, dtype=bool)

    matched_cols = []
    for j, g in enumerate(pred_genes):
        idx = var_upper.get(str(g).strip().upper(), None)
        if idx is not None:
            full_delta[:, idx] = delta_arr[:, j]
            is_pred_gene[idx] = True
            matched_cols.append((str(adata_post.var_names[idx]), j, idx))

    x_post = np.clip(x_pre + full_delta, 0.0, None).astype(np.float32)
    adata_post.layers["pre_expr"] = x_pre
    adata_post.layers["pred_delta"] = full_delta
    adata_post.layers["pred_post_expr"] = x_post
    adata_post.X = x_post
    adata_post.var["is_predicted_gene"] = is_pred_gene
    adata_post.obs["perturbation_label"] = str(pert_label)

    os.makedirs(os.path.dirname(out_h5ad_path), exist_ok=True)
    adata_post.write_h5ad(out_h5ad_path)

    deg_rows = []
    for gname, j_pred, idx_var in matched_cols:
        m_pre = float(np.mean(x_pre[:, idx_var]))
        m_post = float(np.mean(x_post[:, idx_var]))
        m_del = float(np.mean(full_delta[:, idx_var]))
        deg_rows.append({
            "gene": gname,
            "mean_pre": m_pre,
            "mean_pred_post": m_post,
            "pred_delta": m_del,
            "abs_delta": abs(m_del),
        })
    deg_df = pd.DataFrame(deg_rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    if out_deg_csv_path is not None:
        os.makedirs(os.path.dirname(out_deg_csv_path), exist_ok=True)
        deg_df.to_csv(out_deg_csv_path, index=False, encoding="utf-8-sig")

    return adata_post, deg_df


def export_post_perturbation_bulk(
    df_pre: pd.DataFrame,
    pred_delta: np.ndarray,
    pred_genes: list,
    pert_label: str,
    out_post_path: str,
    out_delta_path: str = None,
    out_deg_csv_path: str = None,
) -> tuple:
    """
    Apply predicted perturbation delta to a baseline Bulk expression DataFrame (samples x genes)
    and export predicted post-perturbation expression matrix + DEG summary.
    """
    x_pre_df = df_pre.copy()
    n_samples = len(x_pre_df)
    delta_arr = np.asarray(pred_delta, dtype=np.float32)
    if delta_arr.ndim == 1:
        delta_arr = np.broadcast_to(delta_arr[None, :], (n_samples, len(pred_genes))).copy()

    col_upper = {str(c).strip().upper(): c for c in x_pre_df.columns}
    delta_df = pd.DataFrame(0.0, index=x_pre_df.index, columns=x_pre_df.columns, dtype=np.float32)
    x_post_df = x_pre_df.copy().astype(np.float32)

    deg_rows = []
    for j, g in enumerate(pred_genes):
        orig_c = col_upper.get(str(g).strip().upper(), None)
        if orig_c is not None:
            d_col = delta_arr[:, j]
            delta_df[orig_c] = d_col
            x_post_df[orig_c] = np.clip(x_pre_df[orig_c].values.astype(np.float32) + d_col, 0.0, None)
            m_pre = float(np.mean(x_pre_df[orig_c].values))
            m_post = float(np.mean(x_post_df[orig_c].values))
            m_del = float(np.mean(d_col))
            deg_rows.append({
                "gene": orig_c,
                "perturbation": pert_label,
                "mean_pre": m_pre,
                "mean_pred_post": m_post,
                "pred_delta": m_del,
                "abs_delta": abs(m_del),
            })

    os.makedirs(os.path.dirname(out_post_path), exist_ok=True)
    if out_post_path.endswith(".parquet"):
        x_post_df.to_parquet(out_post_path)
    else:
        x_post_path_csv = out_post_path
        x_post_df.to_csv(x_post_path_csv, encoding="utf-8-sig")

    if out_delta_path is not None:
        os.makedirs(os.path.dirname(out_delta_path), exist_ok=True)
        if out_delta_path.endswith(".parquet"):
            delta_df.to_parquet(out_delta_path)
        else:
            delta_df.to_csv(out_delta_path, encoding="utf-8-sig")

    deg_df = pd.DataFrame(deg_rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    if out_deg_csv_path is not None:
        os.makedirs(os.path.dirname(out_deg_csv_path), exist_ok=True)
        deg_df.to_csv(out_deg_csv_path, index=False, encoding="utf-8-sig")

    return x_post_df, deg_df
