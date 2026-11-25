# -*- coding: utf-8 -*-
import os
import requests
import anndata as ad

DATA_DIR = r"D:\data\AIVC\data"
REC_URL = "https://zenodo.org/api/records/13350497"

TARGETS = [
    "AissaBenevolenskaya2021.h5ad",
    "SrivatsanTrapnell2020_sciplex2.h5ad",
    "DixitRegev2016_K562_TFs_7_days.h5ad",
    "DixitRegev2016_K562_TFs_13_days.h5ad",
    "TianKampmann2019_day7neuron.h5ad",
    "LotfollahiTheis2023.h5ad",
]


def main():
    r = requests.get(REC_URL, timeout=30)
    r.raise_for_status()
    file_map = {f["key"]: f["links"]["self"] for f in r.json().get("files", [])}

    for fname in TARGETS:
        url = file_map.get(fname)
        if not url:
            print(f"[SKIP] Not found on Zenodo: {fname}")
            continue
        out_path = os.path.join(DATA_DIR, fname)
        tmp_path = out_path + ".part"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
            print(f"[EXISTS] {fname} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        else:
            print(f"[DOWNLOAD] {fname} <- {url}")
            with requests.get(url, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)
            print(f"  -> Saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

        # 立即做轻量探查
        a = ad.read_h5ad(out_path, backed="r")
        print(f"  -> shape={a.shape}, obs_cols={list(a.obs.columns)[:12]}")
        for col in ["time", "timepoint", "day", "perturbation", "cell_line", "dose_value"]:
            if col in a.obs.columns:
                vc = a.obs[col].astype(str).value_counts()
                print(f"     {col} (n={vc.shape[0]}): {dict(list(vc.items())[:5])}")
        a.file.close()


if __name__ == "__main__":
    main()
