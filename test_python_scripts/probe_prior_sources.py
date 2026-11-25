import importlib
import gseapy as gp

for pkg in ["omnipath", "decoupler", "networkx", "requests"]:
    try:
        importlib.import_module(pkg)
        print(f"[PKG] {pkg}: installed")
    except Exception:
        print(f"[PKG] {pkg}: NOT installed")

libs = gp.get_library_name(organism="Human")
print(f"[GSEAPY] Total Human libraries: {len(libs)}")
keys = ["kegg", "reactome", "hallmark", "wiki", "trrust", "bioplanet", "encode", "chel", "miRTar", "mirna"]
matched = [l for l in libs if any(k in l.lower() for k in keys)]
for m in matched:
    print("  -", m)
