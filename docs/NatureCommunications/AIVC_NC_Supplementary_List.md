# Supplementary information — Nature Communications submission

Nature Communications requires supplementary figures, small tables and text as **one combined PDF**, with tables longer than one page supplied as separate Excel files. Figure legends stay with the main text; supplementary items are cited as "Supplementary Fig. 1" etc. throughout.

**Numbering rule.** Supplementary items are numbered in order of first citation in the main text. The table below reflects the current manuscript; if any citation is moved during revision, renumber the whole set rather than appending.

---

## 1. Supplementary figures (single PDF, Figs. 1–12)

| Item | Content | Related main item | Source file |
| :--- | :--- | :--- | :--- |
| Supplementary Fig. 1 | Five-member ensemble epistemic calibration and risk–coverage curve | Fig. 2, Table 1 | `test_results/exp34_uq_calibration_plot.png` |
| Supplementary Fig. 2 | Locally gated five-omics translation on blind LUAD and ovarian tumours | Fig. 4c | `test_results/exp27b_local_gated_comparison.png` |
| Supplementary Fig. 3 | Causal signalling topology and blind masked-kinase recovery | Fig. 4c | `test_results/nature_domain_fig2_causal_pathway_net.png` |
| Supplementary Fig. 4 | L1-sparse learned signalling importance and causal regulatory discovery | Table 1, Results | `test_results/exp28_learned_importance_plot.png` |
| Supplementary Fig. 5 | Full 80-pair target-discovery audit (training-free footprint vs cis-mRNA) | Results | `test_results/nature_domain_fig6_target_discovery_3tier.png` |
| Supplementary Fig. 6 | Driver-versus-passenger attribution across 925 multi-gene perturbations | Fig. 3 | `test_results/exp12_driver_vs_passenger_summary.png` |
| Supplementary Fig. 7 | 54-organ GTEx v8 physiological safety atlas | Fig. 6c | `test_results/nature_domain_fig4_human_organ_atlas.png` |
| Supplementary Fig. 8 | Single-cell perturbation manifold, displacement field and marker DotPlot | Fig. 3 | `test_results/nature_domain_fig1_scrna_umap_dotplot.png` |
| Supplementary Fig. 9 | 18-day phase-plane flow, state ribbon and 0–24 h pathway cascade | Fig. 5 | `test_results/nature_domain_fig3_ode_phase_dynamics.png` |
| Supplementary Fig. 10 | DEG volcano, driver waterfall and pathway polarity | Fig. 6 | `test_results/nature_domain_fig5_downstream_volcano_gsea_epistasis.png` |
| Supplementary Fig. 11 | Schematic mechanism diagram, CEBPE+SPI1 (illustrative) | Results | `test_results/exp18_CEBPE_SPI1_sankey_flow.png` |
| Supplementary Fig. 12 | Schematic mechanism diagram, CBL+CNN1 (illustrative) | Results | `test_results/exp18_CBL_CNN1_sankey_flow.png` |

All twelve source files are verified present. They are 300 dpi PNG; Nature Communications prefers vector (`.pdf`, `.svg`, `.eps`) or layered `.tif`/`.psd` at acceptance. Referee-legible resolution is sufficient at first submission.

**Colour check.** Nature Communications requires RGB, avoids red–green contrast, and discourages rainbow scales. Worth an explicit pass over Supplementary Figs. 2, 3, 7 and 10 before submission.

---

## 2. Supplementary Table 1 — accessions and release versions

This table is cited in the Data availability statement and must be complete at submission. Three entries are confirmed from the analysis environment; the remainder must be copied from each dataset's own Data availability statement, not inferred.

| Dataset | Repository | Accession / version | Status | Used for |
| :--- | :--- | :--- | :--- | :--- |
| Replogle Perturb-seq | GEO | see ref. 6 (Data availability) | **fill** | RPE1 benchmark training/evaluation |
| Norman2019 | GEO | **GSE133344** | confirmed | Overexpression and epistasis |
| JoungZhang2023 | GEO | see ref. 11 (Data availability) | **fill** | ORF pretraining |
| MIX-Seq (McFarland 2020) | GEO | see ref. 12 (Data availability) | **fill** | Temporal benchmarks |
| sci-Plex3 (Srivatsan 2020) | GEO | **GSE139944** | confirmed | Zero-fill ablation |
| SchiebingerLander2019 | GEO | see ref. 14 (Data availability) | **fill** | 18-day reprogramming |
| LINCS L1000 | GEO | GSE70138 | confirmed | Chemical scaling |
| Tahoe-100M | Arc Institute | resource portal; cite the Tahoe-100M paper | **fill** | Chemical scaling, pretraining, target set |
| TCGA Pan-Cancer | GDC | *release to be verified* | **fill** | Five-omics, clinical cohorts |
| CPTAC | CPTAC Portal | *release to be verified* | **fill** | Five-omics |
| CCLE / DepMap | Broad | *release to be verified* | **fill** | Five-omics, CRISPR lethality |
| GDSC1/2 | CancerRxGene | *release to be verified* | **fill** | IC₅₀ benchmark |
| PRISM 20Q2 | DepMap | *release to be verified* | **fill** | Viability benchmark |
| DrugComb | DrugComb | *version date to be verified* | **fill** | Synergy benchmark |
| GTEx | GTEx Portal | v8 | confirmed | 54-organ safety |
| EyeGEx | GEO | GSE115828 | confirmed | Retinal safety guard |
| OmniPath | omnipathdb.org | *download date to be verified* | **fill** | Prior atlas |
| BioGRID | thebiogrid.org | *release to be verified* | **fill** | Prior atlas |
| MSigDB / KEGG / Reactome / GO-BP | respective portals | *release to be verified* | **fill** | Prior atlas |

**How to fill.** Two rows are confirmed: GSE133344 (Norman 2019, from the Science paper's Data availability statement) and GSE139944 (Srivatsan 2020, verified against the GEO record, PMID 31806696). Each remaining GEO row names the reference whose Data availability section carries the accession; copy it from there, never from a secondary citation or from memory. Record the release or download date actually used in the analysis, not the current release. For GTEx and TCGA, state the freeze and the pipeline version. Supplementary Figs. 11–12 carry no quantitative claim and are labelled schematic in the main text; they therefore need no Source Data table.

---

## 3. Other required items

1. **Reporting Summary** — required for life-science submissions; generated from the editorial system's questionnaire. Complete it in Adobe Reader and upload as a separate PDF.
2. **Source Data** — 39 CSV files exist under `source_data/` with `SOURCE_DATA_MANIFEST.md`. Nature Communications expects coverage for every figure panel; reorganise by figure (Fig. 1–6, Supplementary Figs. 1–12) before upload.
3. **Code and Software Checklist** — complete for the computational component.
4. **Graphical abstract** — not required by Nature Communications (this was a Cell Systems requirement and has been dropped).
5. **Highlights / eTOC blurb** — not used by Nature Communications; retained only as cover-letter source material in `docs/CellSystems/`.

---

## 4. Submission package checklist

| Item | File | Status |
| :--- | :--- | :--- |
| Manuscript (main text + legends + tables) | `AIVC_NC_Manuscript.md` | ready except author metadata |
| Cover letter | `AIVC_NC_CoverLetter.md` | rewritten to match current framing |
| Supplementary PDF | to be assembled from Figs. 1–12 | pending |
| Supplementary Table 1 | section 2 above | **pending accessions** |
| Source Data | `source_data/` (39 CSV) | reorganise by figure |
| Reporting Summary | editorial system | pending |
| Code archive | GitHub + Zenodo | **pending release** |