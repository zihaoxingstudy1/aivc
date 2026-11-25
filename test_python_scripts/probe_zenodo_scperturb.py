import os
import requests

REC_URL = "https://zenodo.org/api/records/13350497"
DATA_DIR = r"D:\data\AIVC\data"
existing = set(os.listdir(DATA_DIR))

print(f"Querying Zenodo scPerturb record: {REC_URL}")
r = requests.get(REC_URL, timeout=30)
r.raise_for_status()
files = r.json().get("files", [])
print(f"Total files in Zenodo 13350497: {len(files)}")
for f in files:
    key = f["key"]
    sz_mb = f["size"] / 1e6
    if key.endswith(".h5ad"):
        status = "LOCAL" if key in existing else "NEW"
        print(f"  [{status}] {key:45s} {sz_mb:8.1f} MB | {f['links']['self']}")
