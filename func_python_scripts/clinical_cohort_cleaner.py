import os
import re
import gzip
import numpy as np
import pandas as pd


def load_gpl_probe_and_entrez_maps(gpl_dir: str, gpl_ids: tuple = ("GPL96", "GPL97", "GPL570", "GPL571")) -> tuple:
    """
    Load probe_id -> Gene Symbol and Entrez Gene ID -> Gene Symbol mappings from GEO GPL .annot.gz files.
    Returns (probe_to_sym_by_gpl: dict, entrez_to_sym: dict).
    """
    probe_maps = {}
    entrez_to_sym = {}
    for gpl in gpl_ids:
        path = os.path.join(gpl_dir, f"{gpl}.annot.gz")
        if not os.path.exists(path):
            continue
        p2s = {}
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            header = None
            for line in f:
                if line.startswith("!platform_table_begin"):
                    header = next(f).rstrip("\r\n").split("\t")
                    id_idx = header.index("ID")
                    sym_idx = header.index("Gene symbol")
                    gid_idx = header.index("Gene ID") if "Gene ID" in header else None
                    continue
                if header is not None:
                    if line.startswith("!platform_table_end"):
                        break
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) <= max(id_idx, sym_idx):
                        continue
                    probe = parts[id_idx].strip()
                    sym_raw = parts[sym_idx].strip()
                    if not sym_raw or sym_raw in ("---", "NA"):
                        continue
                    sym = sym_raw.split("///")[0].strip().upper()
                    if sym:
                        p2s[probe] = sym
                        if gid_idx is not None and len(parts) > gid_idx:
                            gid_raw = parts[gid_idx].strip().split("///")[0].strip()
                            if gid_raw.isdigit() and gid_raw not in entrez_to_sym:
                                entrez_to_sym[gid_raw] = sym
        probe_maps[gpl] = p2s
    return probe_maps, entrez_to_sym


def load_ensembl_to_sym_from_gtex(gtex_path: str) -> dict:
    """Load Ensembl ID (without version) -> Gene Symbol mapping from GTEx v8 gct.gz."""
    df = pd.read_csv(gtex_path, sep="\t", skiprows=2, usecols=[0, 1])
    ens_to_sym = {}
    for ens, sym in zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)):
        ens_base = ens.split(".")[0].strip()
        sym_clean = sym.strip().upper()
        if ens_base.startswith("ENSG") and sym_clean and sym_clean != "NAN":
            ens_to_sym[ens_base] = sym_clean
    return ens_to_sym


def read_geo_series_matrix_expr(gz_path: str) -> pd.DataFrame:
    """Read expression table from GEO series_matrix.txt.gz as DataFrame (rows=probes, cols=GSM sample_ids)."""
    lines = []
    in_table = False
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                lines.append(line)
    import io
    buf = io.StringIO("".join(lines))
    df = pd.read_csv(buf, sep="\t", index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [str(c).strip('"') for c in df.columns]
    return df


def collapse_and_log2_samples_x_genes(expr_genes_x_samples: pd.DataFrame, id_map: dict = None) -> pd.DataFrame:
    """
    Map row IDs via id_map (if provided), uppercase symbols, average duplicate genes,
    ensure log2 scale, and return DataFrame (rows=samples, cols=genes, float32).
    """
    df = expr_genes_x_samples.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    if id_map is not None:
        mapped = [id_map.get(str(idx).split(".")[0].strip(), "") for idx in df.index]
        df["__gene__"] = mapped
        df = df[df["__gene__"] != ""]
    else:
        df["__gene__"] = [str(idx).strip().upper() for idx in df.index]
        df = df[~df["__gene__"].isin(["", "NAN", "NONE", "---"])]

    df = df.groupby("__gene__", sort=False).mean()
    # Transpose to samples x genes
    mat = df.T.astype(np.float32)
    # Check if linear scale (e.g. MAS5 or raw TPM/FPKM)
    finite_vals = mat.values[np.isfinite(mat.values)]
    if len(finite_vals) > 0:
        q99 = float(np.percentile(finite_vals, 99))
        q50 = float(np.percentile(finite_vals, 50))
        if q99 > 40.0 or q50 > 20.0:
            mat = np.log2(np.clip(mat, 0.0, None) + 1.0).astype(np.float32)
    return mat


def parse_gse9782_raw_chars(gz_path: str) -> pd.DataFrame:
    """Parse GSE9782 where each Sample_characteristics_ch1 cell has 'key = val' in varying positions."""
    sample_ids, titles, platforms = [], [], []
    char_rows = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_platform_id"):
                platforms = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_rows.append([x.strip('"') for x in line.split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break

    records = [{"sample_id": sid, "title": t, "platform_id": p} for sid, t, p in zip(sample_ids, titles, platforms)]
    for row in char_rows:
        for i, cell in enumerate(row):
            if "=" in cell and i < len(records):
                if " = " in cell:
                    k, v = cell.split(" = ", 1)
                else:
                    k, v = cell.rsplit("=", 1)
                k = k.strip().lower().replace(" ", "_")
                records[i][k] = v.strip()
    return pd.DataFrame(records)


def normalize_breast_regimen(raw_str: str) -> str:
    """Map breast cancer regimen strings to canonical semicolon-separated drug names."""
    s = str(raw_str).upper()
    drugs = []
    if any(x in s for x in ["TAXOL", "PACLITAXEL", "T/", "T-", "TFAC", "AC+T", "TCHP", "TONLY", "TH/"]):
        drugs.append("Paclitaxel")
    if "TAXOTERE" in s or "DOCETAXEL" in s or "TAC" in s:
        drugs.append("Docetaxel")
    if any(x in s for x in ["FAC", "AC", "DOXORUBICIN", "ADRIAMYCIN"]):
        drugs.append("Doxorubicin")
    if "FEC" in s or "EPIRUBICIN" in s:
        drugs.append("Epirubicin")
    if any(x in s for x in ["FAC", "FEC", "AC", "CYCLOPHOSPHAMIDE", "TCHP"]):
        drugs.append("Cyclophosphamide")
    if any(x in s for x in ["FAC", "FEC", "5FU", "5-FU", "FLUOROURACIL"]):
        drugs.append("Fluorouracil")
    if "CARBOPLATIN" in s or "TCHP" in s:
        drugs.append("Carboplatin")
    if "TRASTUZUMAB" in s or "TH/" in s or "TCHP" in s:
        drugs.append("Trastuzumab")
    if "PERTUZUMAB" in s or "TCHP" in s:
        drugs.append("Pertuzumab")
    if "IXABEPILONE" in s:
        drugs.append("Ixabepilone")
    if "PEMBROLIZUMAB" in s:
        drugs.append("Pembrolizumab")
    if "NERATINIB" in s:
        drugs.append("Neratinib")
    if "MK-2206" in s:
        drugs.append("MK-2206")
    if "ABT 888" in s or "VELIPARIB" in s:
        drugs.append("Veliparib")
    if "GANITUMAB" in s:
        drugs.append("Ganitumab")
    if "GANETESPIB" in s:
        drugs.append("Ganetespib")
    if "AMG 386" in s:
        drugs.append("Trebananib")
    if not drugs and raw_str and str(raw_str).strip() not in ("", "nan", "None", "none"):
        drugs.append(str(raw_str).strip())
    return ";".join(dict.fromkeys(drugs))
