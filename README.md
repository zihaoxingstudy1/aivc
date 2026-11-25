# AIVC: Mechanistic Multi-Omics & Grey-Box Temporal Virtual Cell World Model

**AIVC** is a unified, mechanism-grounded virtual cell world model for cross-lineage genetic perturbation (CRISPRi/KO), gene overexpression (ORF/CRISPRa), multi-gene combinatorial epistasis, 5-omics protein/phosphorylation translation, continuous-time small-molecule dynamics, and multi-organ + human retina translational safety assessment.

## Key Capabilities
- **Genome-Scale Human Prior Atlas** (`pkb.py`, `pgnn.py`): 3,894 curated pathways, 109,312 OmniPath directed activation/inhibition edges, 721,323 BioGRID physical interactions, and 366,539 pathway cross-talk edges.
- **Cross-Lineage Genetic Knockdown** (`7 lineages -> blind RPE1`, n = 2,016): Top-20 DEG correlation 0.584 versus 0.521 for `Linear-Ridge` (63.8% win rate, `p = 1.0e-43`), leading all nine baselines on Top-20 DEG *r*, direction accuracy, WMSE and perturbation discrimination. Mean *r* is statistically indistinguishable from ridge (0.3965 versus 0.3950); the advantage concentrates on weak-effect perturbations (0.322 versus 0.190).
- **Overexpression & Double-Gene Epistasis** (`JoungZhang2023 -> Norman2019`): #1 across 8-model single-gene overexpression (`Top-20 r = 0.5910`) and 9-model double-gene epistasis (`Mean r = 0.9207`, `Top-20 r = 0.9642`, `Epistasis r = 0.6173`, `57.3%` lower MSE than additive).
- **Masked-Kinase Phospho Recovery & Fast–Slow Grey-Box ODE**: Recovers masked kinase phosphorylation on blind `LUAD/OV` tumors and forecasts continuous `0–24h` and `18-day` trajectories (`State-Anchored ODE`).
- **Zero-Shot Leave-Drug-Out & Safety Guard**: Monotonic scaling laws on `Tahoe-100M` (377 drugs) and `LINCS L1000` (1,794 compounds), paired with `GTEx v8` (54 organs) and `EyeGEx` (453 human retinas) safety profiling.

## Quick Start
```bash
git clone https://github.com/zihaoxingstudy1/aivc.git
cd aivc
pip install -e .
python test_python_scripts/quickstart_verify.py
```
`quickstart_verify.py` runs out-of-the-box without downloading external primary datasets. To run full training and cross-lineage benchmarks on primary datasets (Replogle, Norman, Joung, MIX-Seq, Tahoe-100M, TCGA, CPTAC, GTEx), see `DATA.md` and `REPRODUCE.md` (and run `configure_paths.py --root /path/to/your/aivc` once).

## Package Layout
- `aivc/`: Public Python package entry point (`aivc/__init__.py`) re-exporting core APIs from `func_python_scripts/`.
- `func_python_scripts/`: 23 core model, prior-graph, chemical, multi-omics, and evaluation modules.
- `test_python_scripts/`: Benchmark, ablation, falsification, and figure-generation scripts.
- `source_data/`: 35 plotting Source Data CSVs and manifest.
- `figures/`: Main and supplementary manuscript figures.
- Pretrained AIVC weights (`aivc_5omics_pathway_gnn.pt`, 10 MB) and Source Data workbook are archived on **Zenodo**: https://doi.org/10.5281/zenodo.24539342 (not GitHub).

## Documentation & Source Data
- Manuscript Source: `docs/NatureCommunications/AIVC_NC_Manuscript.md`
- Reproducibility Guide: `REPRODUCE.md`
- Dataset Accessions & Download Guide: `DATA.md`
- Plotting Source Data (35 CSVs): `source_data/SOURCE_DATA_MANIFEST.md`

## Licensing (Academic Non-Commercial Free / Commercial Use Requires Paid License)
AIVC is dual-licensed for free non-commercial academic research and paid commercial licensing (`LICENSE` for code, `MODEL_LICENSE.md` for weights and derivative data):
- **Academic & Non-Commercial Edition (`aivc` Core — Free for Non-Commercial Academic Research)**:
  - **Source Code**: Licensed under the **PolyForm Noncommercial License 1.0.0** (`LICENSE`). Academic, educational, and non-profit research use is free; commercial use is prohibited without a separate paid commercial license.
  - **Model Weights, Outputs & Curated Datasets**: Licensed under **CC BY-NC-SA 4.0** + `MODEL_LICENSE.md` (explicit **Anti-Distillation** & **Sponsor-Penetration** clauses).
- **Commercial Enterprise License (Paid License Required)**: Required for any for-profit biopharma, biotech, AI, CRO, or clinical commercial deployment. Contact `zihaoxing@stu2019.jnu.edu.cn` (details in `MODEL_LICENSE.md`).

## Running the code on your own machine

The analysis scripts were written on the author's workstation and carry absolute
paths of the form `D:\data\AIVC\...`. Before running anything, point them at your own
data root:

```bash
python configure_paths.py --root /path/to/aivc --dry-run   # preview changes
python configure_paths.py --root /path/to/aivc             # rewrite .py files in place
```

The script rewrites both `D:\data\AIVC` and `D:/data/AIVC` occurrences and leaves a
`.bak` copy of every file it touches. If you prefer to keep the originals untouched,
clone twice: edit one clone and keep the other pristine.

No credentials are present in the repository; the tree was scanned before release.
