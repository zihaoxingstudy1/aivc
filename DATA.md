# Data sources and access

No primary data are redistributed in this repository. Several of the source datasets are governed by licences that prohibit redistribution, and the derived predictions that we can redistribute are shipped as Source Data CSVs in `source_data/`.

## What you need before running anything

| Dataset | Role in the paper | Where to obtain it | Accession / version |
| :--- | :--- | :--- | :--- |
| Replogle Perturb-seq | Training lineages and the blind RPE1 benchmark | GEO; see the Data availability section of the original paper | *to be recorded in Supplementary Table 1* |
| Norman 2019 | Overexpression and double-gene epistasis | GEO | GSE133344 |
| Joung & Zhang 2023 | ORF overexpression, equal-data audit | GEO | *to be recorded* |
| MIX-Seq (McFarland 2020) | 6 h / 24 h temporal benchmarks | GEO | *to be recorded* |
| sci-Plex3 (Srivatsan 2020) | Cross-platform zero-fill ablation | GEO | GSE139944 |
| Schiebinger & Lander 2019 | 18-day reprogramming series | GEO | *to be recorded* |
| LINCS L1000 | Chemical scaling, 1,794 compounds | GEO | GSE70138 |
| Tahoe-100M | Chemical pretraining, leave-drug-out scaling | Arc Institute | DOI *to be recorded* |
| TCGA Pan-Cancer | Five-omics and clinical cohorts | GDC | *freeze to be recorded* |
| CPTAC | Proteome and phosphoproteome cohorts | CPTAC Portal | *release to be recorded* |
| CCLE / DepMap | Cell-line multi-omics, CRISPR lethality | Broad DepMap | *release to be recorded* |
| GDSC1/2 | IC₅₀ benchmark | CancerRxGene | *release to be recorded* |
| PRISM 20Q2 | Viability benchmark | DepMap | *to be recorded* |
| DrugComb | Combination synergy | DrugComb portal | *version to be recorded* |
| GTEx | 54-organ safety atlas | GTEx Portal | v8 |
| EyeGEx | Retinal safety guard | GEO | GSE115828 |
| OmniPath | Signed signalling prior | omnipathdb.org | *download date to be recorded* |
| BioGRID | Physical interactions | thebiogrid.org | *release to be recorded* |
| MSigDB / KEGG / Reactome / GO-BP | Pathway sets | respective portals | *release to be recorded* |

**Fill the blanks from the original publication's own Data availability statement, not from a secondary citation.** Record the release actually used in the analysis, not the current release. The completed table becomes Supplementary Table 1.

## Expected local layout

Scripts read from a single data root with this structure:

```
data/
  raw/
    replogle/        norman/        joung/
    mixseq/          sciplex3/      schiebinger/
    lincs/           tahoe/         tcga/
    cptac/           ccle/          gdsc/       prism/
    drugcomb/        gtex/          eyegex/
  priors/
    omnipath_raw_human.tsv
    pathway_catalog.csv
    biogrid_human.tsv
  processed/
```

`data/priors/omnipath_raw_human.tsv` is the one file you must obtain exactly: the atlas construction expects the OmniPath export with `source`, `target`, `sign` and `curation_effort` columns, and the edge counts quoted in the paper (2,838 activation, 1,006 inhibition after curation-effort filtering) are a useful checksum. If your snapshot gives different counts, the snapshot differs.

## Licence note for users

Do not commit any downloaded primary data into a fork of this repository. The `.gitignore` excludes `data/` for this reason. Derived Source Data tables in `source_data/` are released under CC BY-NC-SA 4.0 together with the model weights.