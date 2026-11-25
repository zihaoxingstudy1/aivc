# -*- coding: utf-8 -*-
"""
download_mega_perturbation_datasets.py
使用支持自动断点续传与卡死检测的 curl.exe (--speed-time 30 --speed-limit 10240)
优先下载 Tahoe-100M Pseudobulk + Zenodo 大图谱，再拉取 LINCS L1000。
"""

import os
import subprocess
import requests

BASE_DATA = r"D:\data\AIVC\data"
LINCS_DIR = os.path.join(BASE_DATA, "lincs_l1000")
TAHOE_DIR = os.path.join(BASE_DATA, "tahoe_100m")
os.makedirs(LINCS_DIR, exist_ok=True)
os.makedirs(TAHOE_DIR, exist_ok=True)


def curl_download(url: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"[EXISTS] {os.path.basename(out_path)} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)
        return
    tmp = out_path + ".part"
    print(f"[CURL] {os.path.basename(out_path)} <- {url}", flush=True)
    cmd = [
        "curl.exe", "-L", "-C", "-",
        "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "30",
        "--speed-time", "30", "--speed-limit", "10240",
        "-o", tmp, url,
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
        os.replace(tmp, out_path)
        print(f"  -> Done: {os.path.basename(out_path)} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)
    else:
        print(f"  -> [WARN] Incomplete/stalled ({res.returncode}): {os.path.basename(out_path)}", flush=True)


def main():
    # 1. Tahoe-100M Metadata + Pseudobulk Differential Expression (HuggingFace)
    try:
        from huggingface_hub import snapshot_download
        print("[1/3] Downloading Tahoe-100M metadata & pseudobulk_differential_expression ...", flush=True)
        snapshot_download(
            repo_id="tahoebio/Tahoe-100M",
            repo_type="dataset",
            local_dir=TAHOE_DIR,
            allow_patterns=["metadata/*"],
        )
        print("  -> Tahoe-100M metadata & pseudobulk complete!", flush=True)
    except Exception as e:
        print(f"[WARN] Tahoe-100M failed: {e}", flush=True)

    # 2. scPerturb Large Atlases (Zenodo 13350497)
    try:
        print("[2/3] Querying Zenodo 13350497 large atlases ...", flush=True)
        r = requests.get("https://zenodo.org/api/records/13350497", timeout=30)
        r.raise_for_status()
        fmap = {f["key"]: f["links"]["self"] for f in r.json().get("files", [])}
        for fname in ["SchiebingerLander2019_GSE115943.h5ad", "JoungZhang2023_atlas.h5ad"]:
            if fname in fmap:
                curl_download(fmap[fname], os.path.join(BASE_DATA, fname))
    except Exception as e:
        print(f"[WARN] Zenodo large atlas failed: {e}", flush=True)

    # 3. LINCS L1000 GSE70138 + GSE92742 (NCBI GEO FTP, 30s stall protection)
    print("[3/3] Downloading LINCS L1000 GSE70138 + GSE92742 ...", flush=True)
    lincs_urls = [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_pert_info.txt.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
    ]
    for u in lincs_urls:
        curl_download(u, os.path.join(LINCS_DIR, os.path.basename(u)))


if __name__ == "__main__":
    main()
