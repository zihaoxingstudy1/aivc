import urllib.request, urllib.parse, json, ssl

ctx = ssl.create_default_context()

def esearch_geo(query: str, retmax: int = 8):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({
        "db": "gds",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    })
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    sum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode({
        "db": "gds",
        "id": ",".join(ids),
        "retmode": "json",
    })
    with urllib.request.urlopen(urllib.request.Request(sum_url, headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=20) as r:
        sdata = json.loads(r.read().decode("utf-8"))
    out = []
    for gid in ids:
        rec = sdata.get("result", {}).get(gid, {})
        acc = rec.get("accession", "")
        if acc.startswith("GSE"):
            out.append((acc, rec.get("n_samples", 0), rec.get("title", "")[:110]))
    return out

queries = [
    '("Vorinostat"[Title] OR "Panobinostat"[Title]) AND "Homo sapiens"[Organism] AND gse[Entry Type]',
    '("Trametinib"[Title] OR "JQ1"[Title]) AND "Homo sapiens"[Organism] AND gse[Entry Type]',
    '("TP53 knockout"[Title] OR "EZH2 knockout"[Title] OR "AURKA"[Title]) AND "Homo sapiens"[Organism] AND gse[Entry Type]',
    '("DMSO"[All Fields] AND "Doxorubicin"[Title]) AND "Homo sapiens"[Organism] AND gse[Entry Type]',
]
for q in queries:
    print("\nQUERY:", q)
    for acc, n, title in esearch_geo(q, retmax=8):
        print(f"  {acc} (N={n}): {title}")
