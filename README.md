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
pip install -e .
python test_python_scripts/aivc_flagship_test.py
```

## Documentation & Source Data
- Manuscript Source: `docs/NatureCommunications/AIVC_NC_Manuscript.md`
- Reproducibility Guide: `REPRODUCE.md`
- Dataset Accessions & Download Guide: `DATA.md`
- Plotting Source Data (35 CSVs): `source_data/SOURCE_DATA_MANIFEST.md`

## Licensing (MIT source code / CC BY-NC-SA weights / paid licence for commercial weight use)
AIVC separates source code from trained assets, so the benchmark harness is fully open while commercial use of the weights remains controlled (`LICENSE` for code, `MODEL_LICENSE.md` for weights):
- **Open-Source Academic Edition (`aivc` Core — Free for Non-Commercial Academic Research, Full Top-Journal SOTA Stack)**:
  - **Scope**: Includes **all flagship capabilities** required for top-journal (*Nature / Science / Cell*) reproducibility (`AIVC v1.1 Ensemble`, `Overexpr & Epistasis_Ens5`, `Fast–Slow & State-Anchored ODE`, `5-Omics` masked-kinase recovery, `Tahoe-100M / LINCS L1000` scaling scripts, and `OmniPath` safety analysis), packaged with **clean standardized benchmarks and fast lightweight test data**.
  - **Source Code**: Licensed under the **MIT License** (OSI-approved open source); see `LICENSE`.
  - **Model Weights, Outputs & Curated Datasets**: Licensed under **CC BY-NC-SA 4.0** + `MODEL_LICENSE.md` (explicit **Anti-Distillation** & **Sponsor-Penetration** clauses blocking internal pharma free-riding and commercially funded academic R&D).
- **Commercial Enterprise Pro Edition (Closed-Source — Paid Commercial License Required, Post-Publication Expansion)**: Built atop the full flagship engine for biopharma & clinical hospitals—adding **dirty-data auto-ingestion** (raw `10x` / messy clinical tables / gene alias & batch correction), **larger-scale private fine-tuning**, a **100% commercially compliant `Boltz-2 [MIT] + AutoDock Vina [Apache 2.0]` drug discovery pipeline**, one-click clinical/R&D decision reports, and enterprise Docker/API/UI deployment.
- **Commercial Licensing & 90-Day Evaluation**: terms and contact route are set out in `MODEL_LICENSE.md`.

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
