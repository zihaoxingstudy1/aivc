import os
import gzip
import urllib.request
import urllib.error
import ssl
import pandas as pd
import numpy as np


def get_geo_series_prefix(gse_id: str) -> str:
    """Compute GEO FTP directory prefix, e.g. GSE25066 -> GSE25nnn."""
    num_str = gse_id.replace("GSE", "")
    if len(num_str) <= 3:
        return "GSEnnn"
    return f"GSE{num_str[:-3]}nnn"


def download_file_if_needed(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download a file from URL to dest_path if it does not already exist and is non-empty."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 500
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}")
        if os.path.exists(dest_path) and os.path.getsize(dest_path) < 500:
            os.remove(dest_path)
        return False


def download_geo_series_matrix(gse_id: str, raw_dir: str) -> list:
    """Download series_matrix.txt.gz for a GSE (handles single or multi-platform series)."""
    prefix = get_geo_series_prefix(gse_id)
    base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/"
    cohort_dir = os.path.join(raw_dir, gse_id)
    os.makedirs(cohort_dir, exist_ok=True)

    # First try listing directory to catch multi-platform matrices (e.g., GSE9782-GPL96, GSE9782-GPL97)
    found_files = []
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        import re
        matches = sorted(set(re.findall(r'href="([^"]+_series_matrix\.txt\.gz)"', html)))
        for fname in matches:
            fname_clean = os.path.basename(fname)
            dest = os.path.join(cohort_dir, fname_clean)
            if download_file_if_needed(base_url + fname_clean, dest, timeout=120):
                found_files.append(dest)
    except Exception as e:
        print(f"  [WARN] Listing {base_url} failed ({e}), trying direct filename...")
        fname_clean = f"{gse_id}_series_matrix.txt.gz"
        dest = os.path.join(cohort_dir, fname_clean)
        if download_file_if_needed(base_url + fname_clean, dest, timeout=120):
            found_files.append(dest)
    return found_files


def parse_geo_series_matrix_metadata_and_shape(gz_path: str) -> tuple:
    """
    Parse sample metadata and check expression matrix dimensions from a GEO series_matrix.txt.gz.
    Returns (meta_df, n_genes, first_5_probe_ids, expr_preview_vals).
    """
    sample_ids = []
    titles = []
    platforms = []
    sources = []
    char_rows = []
    in_table = False
    table_header = None
    n_genes = 0
    first_probes = []
    first_vals = []

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_platform_id"):
                platforms = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                sources = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_rows.append([x.strip('"') for x in line.split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                if table_header is None:
                    table_header = [x.strip('"') for x in line.split("\t")]
                else:
                    n_genes += 1
                    if len(first_probes) < 3:
                        parts = line.split("\t")
                        first_probes.append(parts[0].strip('"'))
                        first_vals.append(parts[1:4])

    n_samples = len(sample_ids)
    meta = pd.DataFrame({
        "sample_id": sample_ids,
        "title": titles if len(titles) == n_samples else [""] * n_samples,
        "platform_id": platforms if len(platforms) == n_samples else [""] * n_samples,
        "source": sources if len(sources) == n_samples else [""] * n_samples,
    })

    # Parse key: value pairs from Sample_characteristics_ch1
    parsed_cols = {}
    for idx, row in enumerate(char_rows):
        if len(row) != n_samples:
            continue
        # Detect keys in this row
        keys_in_row = []
        vals_in_row = []
        for cell in row:
            if ":" in cell:
                k, v = cell.split(":", 1)
                keys_in_row.append(k.strip().lower().replace(" ", "_"))
                vals_in_row.append(v.strip())
            else:
                keys_in_row.append(f"char_{idx}")
                vals_in_row.append(cell.strip())
        # Group by key in case a row mixes keys
        unique_keys = set(k for k, v in zip(keys_in_row, vals_in_row) if v != "")
        if len(unique_keys) == 1:
            k_name = list(unique_keys)[0]
            while k_name in parsed_cols:
                k_name = f"{k_name}_{idx}"
            parsed_cols[k_name] = vals_in_row
        else:
            # Row has multiple different keys across samples; store per sample into unified dict
            for s_i, (k_i, v_i) in enumerate(zip(keys_in_row, vals_in_row)):
                if v_i == "":
                    continue
                if k_i not in parsed_cols:
                    parsed_cols[k_i] = [""] * n_samples
                if parsed_cols[k_i][s_i] == "":
                    parsed_cols[k_i][s_i] = v_i
                else:
                    parsed_cols[k_i][s_i] += "; " + v_i

    for k_name, col_vals in parsed_cols.items():
        col_key = k_name
        if col_key in meta.columns:
            col_key = f"char_{col_key}"
        while col_key in meta.columns:
            col_key = f"{col_key}_dup"
        meta[col_key] = col_vals

    return meta, n_genes, first_probes, first_vals
