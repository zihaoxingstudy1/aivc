# Where mechanistic priors work in virtual-cell perturbation prediction and target discovery

**Zihao Xing¹\***

¹ College of Life Science and Technology, Jinan University, Guangzhou 510632, Guangdong, China

\*Correspondence: zihaoxing@stu2019.jnu.edu.cn

---

## Abstract

Predicting how a cell responds to a knockdown, an overexpression or a small molecule is the central test of any virtual-cell model. Large single-cell models interpolate well but fail on unseen cell types, and whole-transcriptome scores conceal this because most genes do not respond. We present AIVC, built on lineage-invariant low-rank programme gating, leave-one-lineage-out ridge anchoring and cross-modal embeddings, with a genome-scale signalling atlas (109,312 directed edges) admitted as a conditional prior. It transfers perturbations across lineages, infers unmeasured phosphosite states and models multi-gene combinations. On 2,016 blind perturbations in a cell type absent from training it recovers differentially expressed genes better than ridge regression (0.584 versus 0.521), gains most where foundation models collapse to chance (weak effects: 0.33 versus 0.19), and an equal-data audit shows the sign of transfer is architectural: the same added data degrades four baselines and improves AIVC. Ablating the prior, or only its sign, leaves accuracy unchanged; the prior does help where the readout is pathway propagation itself, raising masked-kinase phosphosite recovery. Mechanistic priors are therefore task-dependent, not globally beneficial; we map that boundary explicitly.

---

## Introduction

Forecasting how a cell will respond to a gene knockdown, a gene overexpression, a multi-gene combination or a small molecule is the central test of any virtual-cell model, and the practical prerequisite for using such models in target discovery. The last three years have produced large observational representations of single-cell transcriptomes^1–3^, graph-based perturbation predictors^4^ and variational generative models^5^, trained on compendia that now span thousands of perturbations across many lineages^6,7^. These models capture co-expression structure with high fidelity and interpolate well within the lineages and perturbation families on which they were trained.

Extrapolation is a different problem. Rigorous independent evaluation has shown that deep perturbation predictors frequently fail to outperform regularised linear regression when transferred to unseen cell types, and that whole-transcriptome correlation and unweighted mean squared error are dominated by the thousands of genes that do not respond to any single perturbation^8^. Two failure modes follow. Resting transcriptional programmes differ sharply between lineages, so an unconstrained encoder can fit training-lineage background instead of perturbation effect. And a metric that averages over mostly invariant genes rewards predicting "no change" while concealing failure on the genes and perturbations that matter. We refer to these as the baseline gap and the Delta Gap.

Three further limitations constrain translational use. First, signal transduction is executed through protein abundance and post-translational phosphorylation — EGFR pY1068, MAPK pT202/Y204, AKT pS473 — which mRNA-only models cannot recover when the relevant kinase transcript is unmeasured or decoupled from activity^9,10^. Second, loss-of-function screens^6^, gain-of-function screens^7,11^, combinatorial epistasis^7^ and continuous-time drug responses^12–14^ are usually modelled as separate tasks, and pooling incompatible platforms readily produces negative transfer. Third, predictions without calibrated uncertainty, without a normal-tissue safety margin^15^ and without patient-level stratification cannot explain why one patient responds and another does not.

We therefore built AIVC (**Fig. 1**), a virtual-cell model whose primary machinery is lineage-invariant low-rank programme gating coupled to leave-one-lineage-out ridge anchors and cross-modal embeddings, with the curated human signalling atlas^16,17^ admitted as a conditional prior rather than as the backbone. AIVC completes unmeasured protein and phosphosite states, decomposes temporal responses into fast and slow biological timescales, and is evaluated under strict leakage controls, equal-data audits and falsification tests across four perturbation modalities. Because a prior can only help where its edges are informative, we did not assume that it helps: we ablated it, separated its topology from its sign, and report the resulting boundary as a result in its own right.

![Fig. 1](../../test_results/nature_fig0_architecture_schematic.png)

---

## Results

### A genome-scale prior atlas and the question of its contribution

All AIVC modules share one knowledge base, assembled from four resources and aligned to a common gene space: 3,894 curated human pathways from MSigDB Hallmark, KEGG, Reactome and Gene Ontology Biological Process, encoded as row-normalised gene–pathway incidence matrices and retaining pathways with at least three overlapping genes; 109,312 directed, signed OmniPath interactions separated into activation (*A*~pos~) and inhibition (*A*~neg~) operators to give a signed propagation matrix *A*~signed~ = *A*~pos~ − *A*~neg~^16^; 721,323 BioGRID physical interactions providing one-hop neighbourhood masks for local mutation gating^17^; and 366,539 pathway cross-talk edges. Restricted to the 3,603-gene Perturb-seq universe^6^, the atlas resolves 2,838 activation edges, 1,006 inhibition edges, 2,665 active pathways and 577 high-out-degree signalling and kinase hubs. Every downstream module consumes this same signed object. Whether its sign adds information over its unsigned topology is not assumed here but tested directly, and the answer differs between readouts, as the ablation below shows.

### Lineage-invariant transfer closes the baseline and Delta gaps

We trained exclusively on seven human lineages (K562, Jurkat, HepG2, iPSC, neuron, melanoma and A549; 6,918 perturbation profiles) and predicted 2,016 blind, unseen RPE1 perturbations over 3,603 genes (**Table 1**, **Fig. 2**). Global mean correlation barely separates the models: train-mean 0.376, Geneformer-V2 0.391, linear ridge 0.395 and AIVC 0.397. This is the Delta Gap in its purest form, and it is why we do not report mean correlation as a performance claim. On the true differentially expressed genes the ordering changes. Every external deep baseline falls below ridge regression (Top-20 *r* = 0.422–0.474 versus 0.521), whereas the five-member AIVC ensemble reaches 0.584, winning against ridge on 63.8% of unseen perturbations (Wilcoxon *P* = 1.0 × 10⁻⁴³) and against each deep baseline on 72.3–77.9% (*P* < 1 × 10⁻⁵⁰).

Delta-weighted MSE, which weights each gene by its true perturbation magnitude, and the perturbation discrimination score (PDS) remove the contribution of invariant genes. scGPT and Geneformer-V2 sit at chance on PDS (0.5013 and 0.5031), while the AIVC ensemble reaches 0.654 (median 0.772) and the single AIVC model attains the lowest delta-weighted MSE (0.0459; 28.9% below no-change and 14.0% below ridge). The ensemble's Top-1 exact perturbation match rate is 2.18%, forty-four-fold above the 0.05% expected by chance (*P* = 6.14 × 10⁻²⁵ versus scFoundation; *P* = 4.23 × 10⁻⁴⁶ versus ridge).

Three falsification controls support these numbers rather than the reverse. Across five training seeds (42, 101, 202, 303, 404) the single model reproduces at Top-20 *r* = 0.566 ± 0.006, delta-weighted MSE = 0.0479 ± 0.0019 and PDS = 0.657 ± 0.022, between 5.8 and 7.1 standard deviations above ridge. Shuffling training perturbation labels and recomputing the ridge anchors on the shuffled labels collapses discrimination to exactly chance (PDS = 0.4994, MSE = 0.0543), which excludes split leakage as an explanation. Stratifying the 2,016 perturbations into equal tertiles by ‖Δ*y*~true~‖₂ localises the advantage: on strong perturbations ridge is already close to saturation (0.7988 versus 0.8021 for AIVC), whereas on moderate (0.6277 versus 0.5734) and weak perturbations (0.3224 versus 0.1904 for ridge and 0.1028 for scGPT, a 69.3% relative gain) the gated representation retains discrimination. The weakly responding regime, in which signal-to-noise is lowest and black-box models fall to chance, is where a programme-gated representation with an anchored linear backbone still separates perturbations; as the ablation below shows, that advantage does not depend on the signalling prior.

Ensemble disagreement also behaves like an error estimate. Epistemic variance correlates with realised absolute error at Spearman *r* = +0.360 (*P* = 8.17 × 10⁻⁶³; **Supplementary Fig. 1**), and abstaining on the least confident predictions reduces MSE monotonically from 0.01174 at full coverage to 0.00888 at 25% coverage. A model that knows where it is wrong is usable in a screening loop; one that does not is not.

### Mechanistic priors act only where the task is the pathway

Because the prior is the paper's most distinctive ingredient, we tested whether it contributes at all rather than assuming that it does. We rebuilt the causal module with three prior operators on otherwise identical data, splits, ridge anchors, ensemble size and objective: the signed matrix *A*~signed~ = *A*~pos~ − *A*~neg~ used throughout this paper; an unsigned matrix *A*~pos~ + *A*~neg~ that retains topology and edge weight but discards direction; and a zeros operator that removes the prior entirely. All arms were trained with the same five seeds (42, 101, 202, 303, 404).

The signed arm reproduces the reported benchmark exactly, so every comparison is anchored to the published model and not to a re-implementation (**Table 3b**). Discarding the sign does not lower accuracy. Top-20 DEG *r* is 0.584 for signed, 0.587 for unsigned and 0.586 with the prior removed altogether. Restricting to the 613 perturbations (30.4%) that carry at least one OmniPath out-edge does not rescue the prior (signed 0.625, unsigned 0.629, zeros 0.628), and on the weak-effect tertile, where the prior was expected to matter most, removing it is marginally better (0.331 without versus 0.322 with). Seed-to-seed standard deviations for this metric are 0.006–0.012, so none of these differences, in either direction, separates from training noise; PDS and delta-weighted MSE behave identically.

The structural reason is measured rather than inferred. Of the 2,016 blind RPE1 perturbations, 1,403 (69.6%) have no OmniPath out-edge at all, being ribosomal, spliceosomal, metabolic and other housekeeping genes, and the median norm of the causal feature vector across all perturbations is 0.13. The +0.065 Top-20 advantage over ridge (0.586 with every prior edge removed versus 0.521) therefore originates elsewhere in the architecture, and a component-level ablation locates it (**Table 3b**). Removing the control-state encoder — the per-lineage baseline expression vector that conditions every prediction — costs 0.092 Top-20 (0.584 to 0.492; bootstrap 95% CI −0.100 to −0.085), far more than any other single component and almost half again the total advantage over the ridge-only backbone (0.521). Removing the teacher embeddings (co-expression plus ESM-2) costs a further 0.035, of which co-expression alone accounts for 0.009 and ESM-2 alone nothing detectable (CI includes zero). Zeroing every input while retaining the gated ridge backbone leaves 0.445, so the inputs earn +0.139 and the gating earns the remainder. The same ablation explains the weak-effect result: without the control-state encoder, weak-tertile Top-20 falls from 0.322 to 0.213, a 34% loss — twice the average — because weak perturbations are precisely those whose direction and magnitude depend on the target gene's baseline state in the receiving lineage, information that mean-pooled representations discard. The cross-lineage gain is thus attributable to three architectural choices in declining order: a lineage-conditioned linear anchor that absorbs background expression, a control-state encoder that supplies per-perturbation context, and co-expression teacher embeddings; the signalling graph and the ESM-2 protein embedding contribute nothing detectable at transcriptome level.

The same ablation returns the opposite answer when the readout is pathway propagation itself. On blind masked-kinase phosphosite recovery, adding the OmniPath causal term to the five-omics module raises median correlation from 0.110 to 0.164 on TCGA-OV masked kinase sites, from 0.217 to 0.291 on TCGA-LUAD all-phosphosite, and from 0.029 to 0.045 on CPTAC-LUAD masked kinase sites (**Fig. 4c**), while leaving TCGA-LUAD masked sites (0.180 to 0.177) and CPTAC-LUAD all-phosphosite (0.086 to 0.085) unchanged and CPTAC-OV negative. The masked-kinase subsets contain only 7 and 18 evaluable sites, so these are small-sample estimates and we do not treat them as precise.

The two readouts together delimit the prior's domain. In transcriptome-level prediction the target variable is dominated by programmes that are co-expressed with, but not causally downstream of, the perturbed gene, and a signalling prior is diluted below detectability even where edges exist. In phosphosite prediction the target variable is the propagated signal, and the prior contributes. We therefore describe the prior as task-dependent: it is informative when the target variable is pathway state, and not when the target variable merely correlates with it. This is a bounded claim, and it is the claim the data support.

We also verified that the reported benchmark is not a calibration artefact. Because out-of-fold (OOF) predictions were available for every training perturbation, three post-hoc operators — a global output scale, a partial ridge blend and a per-gene shrinkage — were fitted on OOF predictions alone and then applied to the blind set exactly once, with no selection on the test set. No operator improved any headline metric without degrading another: the ridge blend raised mean *r* from 0.397 to 0.423 but cost 0.078 PDS (0.654 to 0.576) and 0.014 Top-20, and the per-gene shrinkage cost 0.031 Top-20 for no compensating gain. The published predictions therefore already sit at the architecture's legitimate optimum, and their systematic amplitude shrinkage (prediction standard deviation 0.30 of truth) does not depress the ranking-based metrics reported here.

### Five-omics completion recovers masked kinase phosphorylation

To connect transcriptome to signalling we assembled 9,950 paired multi-omics profiles: 7,359 TCGA RPPA-plus-RNA tumours^9^, 1,351 CPTAC mass-spectrometry proteomes and phosphoproteomes across ten cancers^10^, and 1,240 CCLE profiles (366 TMT, 874 RPPA). Local one-hop graph gating of somatic mutations (165 hotspot drivers), TF-regulon and miRNA layers, and modality dropout eliminated the cross-cancer negative transfer seen without gating (**Supplementary Fig. 2**), and generalised to tumours held out entirely from training: CPTAC ovarian proteome *r* = 0.356, TCGA-LUAD RPPA phosphosite median *r* = 0.341, CPTAC-LUAD proteome *r* = 0.525.

The decisive test is what remains when the kinase transcript is removed. With all kinase mRNAs set to zero at inference, the cis-mRNA predictor has no evaluable targets and is not reported as a correlation; the informative comparison is therefore AIVC against itself with mRNA visible. Masking reduces median kinase-phosphosite correlation from 0.306 to 0.177 on TCGA-LUAD, from 0.259 to 0.164 on TCGA-OV and from 0.075 to 0.045 on CPTAC-LUAD, so the downstream causal footprint retains 58–63% of the unmasked signal (**Fig. 4c**, **Supplementary Fig. 3**). Within the masked condition the OmniPath causal term is doing measurable work: without it the same masked model reaches only 0.110 on TCGA-OV and 0.029 on CPTAC-LUAD, so the retained fraction falls from 63% to 42% and from 60% to 38% respectively. CPTAC-OV mass-spectrometry phosphoproteomics does not recover under masking (0.025 to −0.014) even with the causal term, which we attribute to the platform's phosphosite coverage and noise floor rather than to the model, and we report it as a negative result.

Two audits bound what this recovery means. Ranking all 2,893 RNA regulators by learned L1-sparse importance (**Supplementary Fig. 4**) enriches approved-drug targets from Tahoe-100M across every cut-off tested: Top-100 odds ratio 3.29 (*P* = 0.026), Top-200 3.58 (*P* = 1.5 × 10⁻³), Top-300 2.88 (*P* = 3.3 × 10⁻³) and Top-500 2.60 (*P* = 2.3 × 10⁻³), with *KIT* first, *PPARG* 27th, *FGFR2* 55th, *FGFR3* 68th and *BCL2* 163rd. Conversely, a training-free linear OmniPath footprint, evaluated on all 80 pre-defined hub–cancer pairs, produced 53/80 positive correlations and 13/80 at uncorrected permutation *P* < 0.05 (4.0 expected by chance), yet 0/80 survived Benjamini–Hochberg control at 3,000 gene-identity permutations, and on the pairs with measurable cis-mRNA the footprint trailed cis-mRNA (median *r* = 0.036 versus 0.237; **Supplementary Fig. 5**). Simple propagation of priors is therefore not sufficient; the gain requires the trained nonlinear multi-omics network.

### Overexpression and combinatorial epistasis under equal-data conditions

Extending from loss to gain of function, we streamed the 1.14-million-cell Joung transcription-factor ORF atlas (485 verified deltas, all test genes excluded)^11^ to pretrain an overexpression module that combines ESM-2 protein embeddings^18^, out-of-fold ridge anchoring, an explicit cis-activation term (α~cis~ = +0.330) and one- and two-hop OmniPath activation cascades. On 21 blind unseen CRISPRa overexpressions (84 training genes)^7^ the module outperforms all seven baselines on Top-20 *r* (0.591 versus 0.510 for ridge, 0.481 for Geneformer-V2, 0.458 for GEARS, 0.300 for scFoundation and 0.287 for scGPT; **Fig. 3a**).

Equal-data auditing changes what this number means, so we report both regimes. With Norman data only (*n* = 84), AIVC ranks first on all six metrics (mean *r* = 0.465, Top-20 *r* = 0.568, delta-weighted MSE = 0.134, PDS = 0.581, versus ridge 0.438 / 0.512 / 0.139 / 0.500). When every baseline is given the same 485 Joung ORFs (*n* = 569), ridge (Top-20 *r* 0.512 → 0.472), static MLP (0.503 → 0.471), Geneformer-V2 (0.437 → 0.411) and scFoundation (0.359 → 0.215) all degrade under the added embryonic-stem-cell lineage shift, whereas AIVC improves to 0.588 (direction accuracy 75.95%, delta-weighted MSE 0.131). The mechanism is architectural: only the lineage-invariant branch — cis-activation plus OmniPath one- and two-hop edges plus the low-rank ESM-2 residual — is updated during pretraining, so H1 background cannot overwrite K562 fine-tuning.

On 27 completely unseen double-gene combinations (104 training pairs; **Table 2**, **Fig. 3b,c**) the additive-anchored ensemble reaches mean *r* = 0.921, Top-20 *r* = 0.964, direction accuracy 0.998 and MSE = 0.00119, 57.3% below the raw additive prediction, with non-additive epistasis residual *r* = 0.617. Because additivity is a strong baseline here, we gave the linear models the identical *d*~A~ + *d*~B~ anchor: a single fitted gain (*s* = 0.830) lowers MSE to 0.00196, and adding a symmetric ESM-2 ridge residual reaches 0.00176 with epistasis *r* = 0.648, marginally above AIVC's 0.617. Even against that equal-input baseline AIVC retains 32.4% lower MSE and higher mean and Top-20 correlations. Beyond pairs, the model generalises from single and double perturbations to unseen triples and higher-order combinations (Top-20 *r* = 0.781 versus 0.751 for ridge) and yields driver-versus-passenger attribution across 925 combinations (**Supplementary Fig. 6**).

### Temporal forecasting and the limits of chemical priors

AIVC's temporal module decomposes latent dynamics into a fast immediate-early state *z*~f~(*t*) and a slow transcriptional state *z*~s~(*t*) coupled through the signed OmniPath operator, integrated with fourth-order Runge–Kutta steps. On 22 unseen cell lines from MIX-Seq^12^ the model reproduces the expected cascade over 0–24 h, suppressing G2-M checkpoint (−0.074) and E2F targets (−0.053) while activating p53 (+0.037), apoptosis (+0.023) and TNF-α/NF-κB signalling (+0.008; **Fig. 5a**).

Absolute accuracy depends on the evaluation subset, and the subset matters more than we initially assumed (**Table 3**). Across all 13 drugs and 335 line–drug pairs at 24 h the fast–slow ODE reaches mean *r* = 0.486, Top-20 *r* = 0.624, direction accuracy 0.835 and MSE = 0.0179, against 0.446 / 0.539 / 0.818 for a static MLP and 0.359 / 0.444 / 0.725 for static linear regression; a joint grey-box variant of the same family reaches 0.535 / 0.647 / 0.883. On the 36 paired 6 h → 24 h trajectories, however, two baselines are competitive or better: a static MLP predicting 6 h directly attains mean *r* = 0.677 and Top-20 *r* = 0.797, above AIVC's 0.579 / 0.631, and a static linear model mapping drug identity to 24 h attains 0.492 / 0.565, above the open-loop rollout's 0.244 / 0.388. This subset is small and dominated by drugs with stereotyped responses, where drug-identity regression is a strong shortcut; the ODE's advantage appears across the full drug set, not in the paired subset. We report the comparison as it stands rather than restricting it to the favourable subset. Against persistence, the advantage is unambiguous: open-loop rollout (Top-20 *r* = 0.388, direction accuracy 0.714, MSE = 0.0278) and Kalman-assimilated rollout (0.382, 0.715, 0.0286) both exceed 6 h persistence (0.219, 0.607, 0.0382), although assimilation with gain 0.1 slightly trails the open loop, indicating that the 6 h observation is already largely contained in the learned dynamics.

Modality alignment, rather than data volume, governs transfer. Zero-filling the 517 genes missing across platforms when sci-Plex3 data are pooled into MIX-Seq^13^ degrades 24 h accuracy from 0.486 to 0.364 on the same architecture, a loss larger than any gain we obtained from additional data. Pretraining a genetic model on chemical Tahoe-100M deltas reduces Top-20 *r* from 0.568 to 0.565 for the matched single model (0.565 for the ensemble variant), whereas same-modality drug pretraining on 365 compounds and 932 pseudobulk profiles improves every metric on the struct-only backbone, at 24 h (mean *r* 0.473 → 0.491, Top-20 0.631 → 0.642, direction accuracy 0.840 → 0.848) and at 6 h (0.546 → 0.556, 0.610 → 0.623, 0.875 → 0.899; **Fig. 4a**). The contrast between the two pretraining directions, on identical data, is the clearest evidence we have that modality match rather than scale determines whether transfer helps.

For zero-shot compound response we fit a closed-form kernel over SMILES, pathway-target and mechanism-of-action views. Both benchmark sets show monotonic error reduction with training-set size and a discrimination ceiling. On Tahoe-100M, MSE falls from 0.178 at 30 drugs to 0.171 at 301 drugs, and the full kernel attains Top-20 *r* = 0.693 across 76 unseen compounds (**Fig. 4b**) — but a single-view SMILES kernel reaches 0.689 with lower MSE (0.170), and train-mean reaches 0.685. On 1,794 LINCS L1000 compounds (1,435 training, 359 unseen)^19^ MSE falls monotonically from 0.952 at 50 drugs to 0.882 at 1,435, the lowest of any model tested (ridge 0.891, train-mean 0.945), yet Top-20 *r* falls slightly from 0.653 to 0.651 and stays below ridge (0.675) and train-mean (0.683) at every training size (**Fig. 5c**). Chemical priors of this kind improve error magnitude and data efficiency; they do not deliver zero-shot discrimination at these sample sizes. We regard that boundary as a result, not a shortfall: it identifies where a mechanistic model must be used and where a similarity kernel is sufficient.

On 18-day mouse iPSC reprogramming (259,155 cells)^14^, forecasting blind days D12.5–D18 from D0–D12, the state-anchored ODE matches persistence from D12 almost exactly (mean *r* = 0.8743 versus 0.8740; Top-20 0.8808 versus 0.8807; MSE 0.0978 versus 0.0982), while both clearly exceed open-loop D0 rollout (0.721) and a time-conditioned static MLP (0.619; **Fig. 5b**). A 0.0003 difference is not a win, and we treat the 18-day horizon as persistence-limited: anchoring recovers everything the trajectory has already revealed and extrapolates almost nothing beyond it.

### Pan-cancer pharmacogenomics and clinical cohorts

At the phenotypic level the same representations map to viability and combination response. AIVC-Synergy covers GDSC1/2 (450,000 compound–cell-line pairs across 1,898 lines)^20^, PRISM 20Q2 (739,000 viability profiles × 1,488 compounds)^21^ and DepMap CRISPR lethality. Transferring genetic double-target epistasis priors into a shared ESM-2 target space improved symmetric drug-combination synergy prediction on DrugComb v1.5 (342,000 pairs)^22^ and Sanger GDSC combinations (72,000 pairs) from ROC-AUC 0.8349 (PR-AUC 0.3119) to 0.8389 (PR-AUC 0.3267). The gain is modest in absolute terms and we do not present it as a step change in synergy prediction.

Across 17 curated clinical treatment cohorts (5,265 patients) we performed four-arm intrinsic-versus-microenvironment ablations (**Fig. 6a**), paired tumour-killing validation and external bulk RNA-seq consistency checks. In 531 chemotherapy-resistant breast cancers we screened 1,770 dual-drug combinations against an eight-organ toxicity Pareto frontier (**Fig. 6b**), and paired every prediction with 54-organ GTEx v8 and EyeGEx safety margins (**Fig. 6c**)^15^. One result deserves emphasis because it constrains the clinical claim: none of the 1,770 pairs achieved a strictly positive absolute therapeutic window (maximum −0.646, median −3.423). Combination benefit, where it exists, is relative rather than absolute at the concentrations modelled, so the appropriate use of this screen is ranking and dose scheduling, not binary nomination.

### A translational case: EGFR-driven lung adenocarcinoma and T790M resistance

We traced one clinical problem — *EGFR*-driven lung adenocarcinoma with acquired T790M resistance — through all four layers. In blind LUAD phosphosite prediction, EGFR pY1068 is among the three best-recovered sites (*r* = 0.524), together with SHP2 pY542 (0.490), MAPK pT202/Y204 (0.467) and AKT pS473 (0.422), and L1-sparse gating places the EGFR ligand *EREG* at rank 22 of 2,893. Under complete kinase-mRNA masking the model still recovers EGFR-axis phosphorylation through the 315 *EGFR* edges feeding *BRAF* and *MAP2K1* (TCGA-LUAD *r* = 0.177, all-phosphosite 0.291; CPTAC-LUAD 0.045).

Target-matched retrieval against *EGFR* annotations with RDKit druglikeness filtering returns approved, PAINS-free inhibitors including gefitinib (QED 0.518, 446.91 Da, zero Lipinski violations), afatinib (0.457) and osimertinib. Because AIVC operates at the cellular rather than atomic level, we used third-party co-folding and quantum rescoring strictly as an orthogonal check: gefitinib falls from 73.25% to 45.02% binding probability on the T790M gatekeeper, whereas osimertinib rises from 84.86% to 87.13% with quantum binding energy improving from −56.75 to −60.62 kcal mol⁻¹. Osimertinib is under-ranked by 2D QED (0.182) because of its covalent warhead and mesylate salt, which is exactly the failure mode a structural layer is meant to catch.

The safety layer separates classes cleanly. Across 54 GTEx organs, six targeted kinase inhibitors average ln(IC₅₀) = +3.41 (range +2.46 to +4.83), against +1.71 (−3.87 to +5.91) for twelve cytotoxics: osimertinib averages +2.46 (whole blood +1.80, heart +2.73, brain +3.01, lung +2.51) and gefitinib +2.76, whereas docetaxel averages −1.84 and bortezomib −3.87 (**Supplementary Fig. 7**). These targets are expressed in 52–54 of 54 normal organs, so the margin comes from potency relative to effective exposure, not from restricted expression; the same breadth explains the on-target toxicities seen clinically. Patient heterogeneity follows from the same decomposition: top versus bottom causal-footprint quartiles differ sharply in measured target phosphorylation (BRCA STAT3 pY705 BH-FDR = 6.44 × 10⁻¹⁵; BLCA STAT3 6.40 × 10⁻¹¹; KIRC HER2 pY1248 7.78 × 10⁻¹⁰), although the permutation-based FDR for these quartile contrasts is 0.053–0.077 and we report it as suggestive rather than definitive.

### Downstream mechanistic visualisation

To make predictions usable in downstream workflows we provide a suite: single-cell manifolds and perturbation displacement fields (**Supplementary Fig. 8**), phase-plane and state-ribbon dynamics (**Supplementary Fig. 9**), volcano and driver-attribution waterfalls (**Supplementary Fig. 10**), and two schematic mechanism diagrams for CBL–CNN1 and CEBPE–SPI1 (**Supplementary Figs. 11, 12**), illustrations rather than quantitative evidence, reusing the causal topology (**Supplementary Fig. 3**) and the 54-organ safety atlas (**Supplementary Fig. 7**) introduced above. Computed panels derive from the same latent states as the benchmarks and are diagnostic rather than decorative.

---

## Discussion

The strongest results here are architectural, and they hold under deliberately unfavourable conditions. Under leave-one-lineage-out transfer to a cell type absent from training, AIVC raises top DEG recovery over ridge by 12.1% relative (0.584 versus 0.521; 63.8% win rate), cuts delta-weighted error by 14%, and lifts perturbation discrimination from chance to 0.65. The advantage is largest where it is hardest: on weak-effect perturbations, where foundation-model discrimination collapses to chance, the gated anchor retains 0.33 versus 0.19 for ridge. Most tellingly, the equal-data audit — adding the same 485 held-out perturbations to every model's training set — degrades four baselines and improves AIVC on the same data, which is direct evidence that the sign of transfer is an architectural property rather than a data-volume artefact. Calibration is also real: uncertainty estimates rank prediction error (Spearman 0.360, *P* = 8 × 10^-63^), so the model can abstain when it is unreliable.

Recent benchmark work shows that parameter count and observational scale do not by themselves buy out-of-distribution generalisation^8^. Our results say where generalisation does come from, and the answer is narrower than expected. What closes the baseline and Delta gaps is not the mechanistic prior: ablating it, or only its sign, leaves transcriptome-level accuracy unchanged even on subsets that carry signalling edges. What closes the gaps is the decomposition the component ablation identifies: a leave-one-lineage-out anchor that absorbs lineage background so the nonlinear branch learns perturbation effect rather than training-lineage context, a control-state encoder that supplies the per-perturbation context weak effects depend on, and co-expression teacher embeddings. The prior earns its place in a different regime — where the predicted variable is the propagated signal itself.

Three observations define that boundary. First, strong perturbations are already well approximated by linear projections of gene programmes (ridge Top-20 *r* = 0.799 versus 0.802), and the advantage concentrates on moderate and weak perturbations, where foundation models collapse to chance. The gated representation adds not general predictive power but the structure a low signal-to-noise regime requires: a low-rank programme basis anchored per lineage. Second, the equal-data audit on overexpression shows why naive pooling fails and why architecture determines the sign of transfer: the same 485 hESC ORFs degrade four baselines and improve AIVC, because only the lineage-invariant branch is updated. Third, the 80-pair audit shows that training-free propagation of the same prior fails strict permutation control and trails cis-mRNA where mRNA exists, which is consistent with the ablation: the atlas alone carries little information, and what value it has is realised only inside a trained nonlinear module and only on pathway-level readouts.

These results converge on the paper's most transferable finding: a mechanistic prior contributes in proportion to how much of the target variable is the pathway itself. Where the readout is a phosphosite whose abundance is set by kinase activity, the prior moves the estimate (0.110 to 0.164 on masked TCGA-OV kinase sites, 0.029 to 0.045 on CPTAC-LUAD). Where the readout is a whole-transcriptome delta in which pathway targets are a minority among co-expressed programmes, the prior is diluted below detection even on perturbations that have edges (0.625 signed versus 0.628 with no prior). This is a scope condition rather than a failed ingredient, and it makes the intended use sharper: use a prior to predict pathway state, and use a gated anchor to predict transcriptional consequence.

The deliverable is a working, falsified boundary, not a leaderboard claim; the model is usable today for three screening tasks: ranking weak-effect genetic perturbations across lineages, recovering kinase pathway activity from transcriptomes when the phosphosite was never measured, and deprioritising unsafe combinations using a 54-tissue expression margin. Each claim carries a quantitative confidence statement and each failure mode is reported alongside — the standard we would like virtual-cell evaluation to adopt.

Negative results are as informative as positive ones and stay in the main text. On the 36 paired temporal trajectories, static baselines match or beat the ODE; at 18 days the anchored model is indistinguishable from persistence; and for zero-shot compounds a single-view SMILES kernel matches the multi-view kernel on Top-20 correlation and beats it on MSE. The model's competence is therefore cross-lineage genetic perturbation, especially of weak effect, phosphosite recovery when the kinase transcript is absent, and ranking under a safety constraint; general temporal extrapolation and zero-shot chemical discrimination are not supported.

Five limitations follow directly. Our temporal validations use longitudinal *in vitro* series rather than *in vivo* pharmacokinetic/pharmacodynamic sampling, where clearance and systemic immunity add timescales the model does not represent. The model does not predict high-content imaging morphology, and aligning the latent state to Cell Painting readouts is the natural multimodal extension. Prospective validation of uncertainty-ranked combinations in patient-derived organoids has not been performed, and until it is, the clinical outputs should be read as prioritisation rather than recommendation. The masked-kinase evidence for the task-dependence rule rests on 7 and 18 evaluable phosphosites in the two most favourable cohorts, which is enough to establish direction but not to estimate an effect size; a phosphoproteomic cohort with broader site coverage would test the rule properly. Finally, several translational numbers rest on cohorts with platform-specific phosphosite coverage; the CPTAC-OV negative result reminds us that cross-platform phosphoproteomics is a measurement problem before it is a modelling problem.

---

## Methods

### Signalling prior atlas

Pathways were compiled from MSigDB Hallmark, KEGG, Reactome and GO Biological Process and encoded as row-normalised gene-to-pathway (*H*~g→p~) and pathway-to-gene (*H*~p→g~) bipartite matrices, retaining pathways with at least three overlapping genes. OmniPath interactions were split into row-normalised activation (*A*~pos~) and inhibition (*A*~neg~) adjacency matrices, giving the signed operator *A*~signed~ = *A*~pos~ − *A*~neg~. BioGRID physical interactions were aligned to the same gene space to construct binary one-hop masks *M*~1hop~ used for local mutation gating. All matrices were restricted to the 3,603-gene Perturb-seq universe before any model fitting, so no test-set gene identity enters atlas construction. Pathway cross-talk edges were generated by intersecting gene sets of pathway pairs at Jaccard ≥ 0.1 and counting shared signed intermediates.

### Cross-lineage causal perturbation model

Given multi-lineage training deltas *Y*~tr~ ∈ ℝ^{N×G}, we extract a rank-*K* gene-program basis *U*~prog~ ∈ ℝ^{K×G} (*K* = 64) by truncated SVD and precompute lineage-invariant one- and two-hop causal footprints *F*~causal~ = (*A*~signed~ + 0.5 *A*~signed~²) *U*~prog~ᵀ. For target gene *p* in lineage *c*:

$$\hat{\Delta y}_{c,p} = \left(1 + 0.5\tanh(g_\theta(s_{c,p}))\right)\cdot(\alpha\hat{y}^{\text{ridge}}_p) + 0.5\,h_\phi(x^{\text{ctrl}}_c, e_p, f^{\text{causal}}_p, s_{c,p})\,U_{\text{prog}}$$

where *s*~c,p~ = [*x*~ctrl~, *z*~ctrl~, *p̂*~5Omics~, *φ̂*~5Omics~, log(1 + ‖*ŷ*~ridge~‖₂)] injects baseline protein and phosphosite states predicted by the five-omics module, and *e*~p~ concatenates co-expression and ESM-2 (150M) embeddings. Ridge anchors are computed under leave-one-lineage-out cross-validation with α = 50.0, so the anchor for a held-out lineage never sees that lineage. Models were trained with AdamW (learning rate 1 × 10⁻³, weight decay 1 × 10⁻³, batch size 256, 45 epochs) under a composite objective ℒ = MSE(Δ*y*, Δ*y*) + 0.05(1 − cos(Δ*y*, Δ*y*)). A five-member deep ensemble (seeds 42, 101, 202, 303, 404) provides both the ensemble prediction and the epistemic scale σ~ens~ = *G*⁻¹ Σ~j~ sd~m~(Δ*y*~j~^(m)^). The prior ablation replaces *A*~signed~ inside *F*~causal~ with a single alternative operator while holding data, lineage splits, ridge anchors, ensemble size, seeds and objective fixed: *A*~unsigned~ = *A*~pos~ + *A*~neg~, which preserves topology and total edge weight but discards direction, and *A*~zeros~ = 0, which removes the prior term. The signed arm reproduces the reported benchmark exactly, so all three arms are directly comparable.

### Five-omics completion and kinase masking

Paired profiles across TCGA, CPTAC and CCLE were harmonised onto 2,893 shared RNA regulators, 1,200 mass-spectrometry proteome targets, 800 phosphosites and 217 RPPA antibodies, with LUAD and ovarian tumours held out entirely. The pathway network combines cis-RNA skip connections, BioGRID one-hop locally gated mutation modulation (*W*~mut~ ⊙ *M*~1hop~ over 165 hotspot drivers), TF-regulon and KEGG message passing, miRNA encoders and L1-sparse regulator attention with modality dropout. Under masking (*x*~rna~[:, kinase_mask] = 0) a signed one- and two-hop downstream footprint residual adapter ([*X A*ᵀ, *X A*, *X* (*A*ᵀ)²]), trained strictly on seen cancers, recovers masked kinase phosphorylation.

Two properties of this evaluation should be stated explicitly. Masking removes every measurable cis-mRNA target for the masked kinases, so a cis-mRNA predictor cannot be scored on the masked subset; we therefore report AIVC with and without masking and do not impute a correlation for an empty target set. Correlations computed on fewer than three targets are marked not evaluable throughout. Phosphosite correlations use the median across sites, because platform coverage differs by an order of magnitude between RPPA (tens to hundreds of sites) and mass spectrometry (up to 800).

### Overexpression and combinatorial epistasis

For unseen single-gene overexpression:

$$\hat{\Delta y}_p = \alpha_r \hat{y}^{\text{ridge}}_p + \alpha_{\text{cis}}\mathbf{e}_p + \alpha_1 A_{\text{signed}}[:,p] + \alpha_2 A_{\text{signed}}^2[:,p] + 0.15\,h_\psi(e^{\text{ESM2}}_p, f^{\text{causal}}_p)U_{\text{prog}}$$

with *K* = 32. During same-modality pretraining on the 485 zero-leakage Joung ORFs (all 21 Norman test genes excluded), only the lineage-invariant mechanistic branch — cis-activation, one- and two-hop OmniPath terms and the low-rank ESM-2 residual — is updated; the ridge anchor and output layer are frozen, which is why added hESC data cannot distort downstream K562 fine-tuning. For unseen pairs (*A*, *B*), the epistasis module anchors on the additive (*d*~A~ + *d*~B~) and dominant envelopes, adds OmniPath cross-talk (*A*~signed~[:, *A*] ⊙ *A*~signed~[:, *B*]) and symmetric bilinear terms over latent and ESM-2 embeddings. Equal-input baselines receive the identical *d*~A~ + *d*~B~ anchor, a global gain *s* fitted by least squares on the 104 training pairs (*s* = 0.8295), and ridge residual regression over α ∈ {10, 100, 1000}.

### Fast–slow grey-box ODE

Latent fast (*z*~f~) and slow (*z*~s~) states are integrated with fourth-order Runge–Kutta steps:

$$\frac{dz_f}{dt} = -\gamma_f z_f + u_f e^{-k_{\text{dec}}t}, \qquad \frac{dz_s}{dt} = -\gamma_s z_s + 0.5\tanh(z_f W_{f\to s}^\top + z_s W_{s\to s}^\top) + 0.4u_s$$

with coupling matrices anchored on *U*~prog~ *A*~signed~ *U*~prog~ᵀ and Δ̂*y*(*t*) = (*z*~f~(*t*) + *z*~s~(*t*)) *U*~prog~. Intermediate observations are assimilated with Kalman gain *K* = 0.1; long-horizon trajectories are anchored on the latest observed state (D12 for the 18-day series). Two instantiations of this family appear in the benchmarks and are labelled wherever they are compared: the fast–slow model used for the primary 24 h and 6 h results, and a joint grey-box variant reported alongside it. Ablations are always computed within one instantiation; where a comparison crosses instantiations (the struct-only pretraining backbone) we say so and restrict the claim to the within-backbone contrast.

### Multi-view chemical kernel

For zero-shot leave-drug-out benchmarks (Tahoe-100M 301 → 76 drugs; LINCS L1000 GSE70138 1,435 → 359 drugs), *K* = *w*₁*K*~SMILES~ + *w*₂*K*~Pathway+Target~ + *w*₃*K*~MoA~ is solved in closed form as kernel ridge regression. We report MSE and Top-20 correlation separately because they diverge: error magnitude improves monotonically with training size, whereas discrimination saturates at or below trivial baselines, and we do not describe the latter as a scaling law.

### Benchmark models and probe protocol

GEARS and scVI were trained with their official implementations on identical splits. For scGPT, Geneformer-V2-104M and scFoundation-100M we performed frozen-backbone feature probing: pretrained 512- or 768-dimensional gene embeddings with a matched two-layer MLP readout, AdamW, identical splits and matched optimisation budgets. Probe-only adaptation is a conservative choice that understates what full fine-tuning might achieve; we did not run full fine-tuning, and the baseline comparison should be read with that asymmetry in mind. All AIVC models were trained from raw public data and prior graphs without distillation from any external model. Structural verification used Boltz-2 co-folding and MACE-OFF23 rescoring purely as third-party downstream checks on EGFR WT versus T790M complexes; no AIVC parameter was fitted on structural data.

### Calibrated metrics and falsification tests

Delta-weighted MSE weights each gene by its true perturbation magnitude: WMSE~i~ = Σ~g~ *w*~i,g~(Δ̂*y*~i,g~ − Δ*y*~i,g~)², with *w*~i,g~ = |Δ*y*~i,g~| / Σ~g'~|Δ*y*~i,g'~|. The perturbation discrimination score ranks each prediction against all others by cosine similarity to the true delta, PDS~i~ = 1 − rank~j~(*D*~ij~)/(*N* − 1), using mid-rank tie-breaking so that constant or zero predictions yield exactly 0.500. Target enrichment uses one-sided Fisher exact tests; the 80-pair audit uses 3,000 gene-identity permutations with Benjamini–Hochberg control. Three falsification controls are reported: five-seed retraining, label permutation with recomputed ridge anchors, and effect-size tertile stratification.

### Statistics and reproducibility

All analyses are deterministic reanalyses of public data with fixed, pre-generated splits; no randomisation or blinding applies, and no outlier removal was performed. The unit of analysis is stated wherever a number is reported: perturbations (*n* = 2,016 in the RPE1 benchmark; 21 single-gene overexpressions; 27 unseen pairs), line–drug pairs (*n* = 335 for the 13-drug 24 h set; *n* = 36 for the paired 6 h → 24 h subset), tumours (TCGA *n* = 7,359; CPTAC *n* = 1,351), patients (*n* = 5,265 across 17 cohorts) or cells (*n* = 259,155 for the reprogramming series). Pairwise model comparisons use two-sided paired Wilcoxon signed-rank tests, chosen because pairings are exact by construction (same perturbation, same lineage) and because no distributional assumption is warranted across perturbations; exact *P* values are reported rather than thresholded, and test statistics and per-perturbation values are provided as Source Data. No parametric test (t-test or ANOVA) is used anywhere in the paper, so *t*, *F* and degrees of freedom are not applicable. Dispersion is reported as standard deviation across five training seeds where seeds vary, and as 1,000-resample bootstrap 95% confidence intervals where the quantity is a model-level mean. Error bars in all figures are defined in the corresponding legend. Multiple comparisons are controlled within each audit family by Benjamini–Hochberg; nominal permutation *P* values are labelled uncorrected where shown. Sample sizes were fixed by dataset availability rather than by power calculation, and all perturbations, tumours and drugs passing the stated quality filters were analysed. Post-hoc calibration operators (global scale, ridge blend, per-gene shrinkage) were fitted exclusively on out-of-fold predictions from seven leave-one-lineage-out folds over the 6,918 training perturbations, frozen, and applied to the blind set exactly once; no operator was selected, tuned or iterated on the blind set. Replication is computational: every benchmark was re-run from the released code and splits to confirm the reported values, and the five-seed arms provide the reproducibility estimate for the stochastic components.

---

## Data availability

All primary datasets are public and were obtained from their original repositories: Replogle Perturb-seq, Norman2019, JoungZhang2023, MIX-Seq, sci-Plex3, SchiebingerLander2019, LINCS L1000 (GSE70138), Tahoe-100M, TCGA, CPTAC, CCLE/DepMap, GDSC1/2, PRISM 20Q2, DrugComb, GTEx v8, EyeGEx (GSE115828), OmniPath and BioGRID. Accessions and release versions are listed in Supplementary Table 1. Derived data supporting every figure — per-perturbation predictions, bootstrap intervals, audit tables and the 35 Source Data CSV files indexed in `SOURCE_DATA_MANIFEST.md` — are provided with this paper at https://github.com/zihaoxingstudy1/aivc (archived on Zenodo upon release). Source data are provided with this paper.

## Code availability

The AIVC package (`aivc` 1.1.0), prior-graph builders, benchmark harness and falsification scripts are released under the MIT License at https://github.com/zihaoxingstudy1/aivc (tag `v1.1.0`). Trained model weights and curated derivative datasets are released under CC BY-NC-SA 4.0 with supplementary terms in `MODEL_LICENSE.md`; commercial use of the weights requires a separate licence. The MIT licence applies to all source code in the repository.

---

## References

1. Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nat. Methods* **21**, 1470–1480 (2024).
2. Theodoris, C. V. et al. Transfer learning enables predictions in network biology. *Nature* **618**, 616–624 (2023).
3. Hao, M. et al. Large-scale foundation model on single-cell transcriptomics. *Nat. Methods* **21**, 1481–1491 (2024).
4. Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nat. Biotechnol.* **42**, 927–935 (2024).
5. Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. & Yosef, N. Deep generative modeling for single-cell transcriptomics. *Nat. Methods* **15**, 1053–1058 (2018).
6. Replogle, J. M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* **185**, 2559–2575 (2022).
7. Norman, T. M. et al. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* **365**, 786–793 (2019).
8. Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat. Methods* https://doi.org/10.1038/s41592-025-02772-6 (2025).
9. Weinstein, J. N. et al. The Cancer Genome Atlas Pan-Cancer analysis project. *Nat. Genet.* **45**, 1113–1120 (2013).
10. Edwards, N. J. et al. The CPTAC data portal: a resource for cancer proteomics research. *J. Proteome Res.* **14**, 2707–2713 (2015).
11. Joung, J. et al. A transcription factor atlas of directed differentiation. *Cell* **186**, 209–229 (2023).
12. McFarland, J. M. et al. Multiplexed single-cell transcriptional response profiling to define cancer vulnerabilities and therapeutic mechanism of action. *Nat. Commun.* **11**, 4296 (2020).
13. Srivatsan, S. R. et al. Massively multiplex chemical transcriptomics at single-cell resolution. *Science* **367**, 45–51 (2020).
14. Schiebinger, G. et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell* **176**, 928–943 (2019).
15. GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* **369**, 1318–1330 (2020).
16. Türei, D., Valdeolivas, A., Gul, L., Palacios, N., Cesareni, G. & Saez-Rodriguez, J. Integrated intra- and intercellular signaling knowledge for multicellular omics analysis. *Mol. Syst. Biol.* **17**, e9923 (2021).
17. Oughtred, R. et al. The BioGRID database: a comprehensive biomedical resource of curated protein, genetic, and chemical interactions. *Protein Sci.* **30**, 187–200 (2021).
18. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023).
19. Subramanian, A. et al. A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. *Cell* **171**, 1437–1452 (2017).
20. Yang, W. et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. *Nucleic Acids Res.* **41**, D955–D961 (2013).
21. Corsello, S. M. et al. Discovering the anticancer potential of non-oncology drugs by systematic viability profiling. *Nat. Cancer* **1**, 235–248 (2020).
22. Zagidullin, B. et al. DrugComb: an integrative cancer drug combination data portal. *Nucleic Acids Res.* **47**, W43–W51 (2019).

---

## Acknowledgements

We thank the open-source single-cell, multi-omics and systems-biology communities for making perturbation atlases, proteogenomic cohorts and curated prior databases freely accessible. This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors; all computation was performed on the authors' own resources.

## Author contributions

Zihao Xing conceived the study, designed and implemented the AIVC architecture, performed all computational benchmarks and falsification audits, generated the figures and wrote the manuscript.

## Competing interests

The author holds intellectual property rights in the AIVC platform. Source code is released under the MIT License; trained model weights and curated derivative datasets are released under CC BY-NC-SA 4.0 with supplementary terms, and commercial use of the weights requires a separate paid licence.

## Ethics declarations

This study reanalyses publicly available, de-identified human data (TCGA, CPTAC, GTEx, EyeGEx) and public cell-line datasets; no new human participants or animals were involved, and no ethical approval was required.

## Materials and correspondence

Requests for materials and further information should be addressed to Zihao Xing (zihaoxing@stu2019.jnu.edu.cn).

## Figure legends

**Fig. 1 | AIVC architecture.** A genome-scale human molecular prior atlas (signed OmniPath activation and inhibition edges, BioGRID physical interactions, pathway cross-talk) feeds five coupled modules: lineage-invariant causal perturbation transfer (AIVC-Causal); dual-track five-omics protein and phosphosite completion (AIVC-5Omics); gene overexpression and combinatorial epistasis modelling (AIVC-Overexpr, AIVC-Epistasis); fast–slow grey-box temporal ODE forecasting (AIVC-ODE); and calibrated deep-ensemble uncertainty paired with 54-organ GTEx safety profiling.

![Fig. 2](../../test_results/nature_fig1_sota_multipanel.png)

**Fig. 2 | Cross-lineage out-of-distribution benchmark.** **a**, Per-perturbation Top-20 differentially expressed gene correlation across ten models on 2,016 blind unseen RPE1 perturbations, with paired Wilcoxon brackets. **b**, Paired hexbin density comparing the AIVC ensemble with ridge regression (63.8% win rate, *P* = 1.0 × 10⁻⁴³). **c**, Bootstrap 95% confidence intervals for all ten models. **d**, Normalised multi-metric radar profile.

![Fig. 3](../../test_results/nature_fig2_overexpr_and_epistasis.png)

**Fig. 3 | Overexpression and combinatorial epistasis on Norman2019.** **a**, Lollipop gain chart for 21 unseen single-gene CRISPRa overexpressions. **b**, Multi-metric profile across 27 unseen double-gene combinations. **c**, Pareto frontier of MSE versus non-additive epistasis residual *r*, bubble area scaled by Top-20 correlation.

![Fig. 4](../../test_results/nature_fig3_drug_temporal_scaling.png)

**Fig. 4 | Pharmacological pretraining, zero-shot chemical scaling and masked-kinase recovery.** **a**, Paired slopegraph of 6 h and 24 h metrics on 22 unseen MIX-Seq cell lines after same-modality Tahoe-100M pretraining. **b**, Leave-drug-out data scaling across 76 unseen Tahoe-100M compounds. **c**, Cross-cohort matrix of blind masked-kinase phosphosite recovery.

![Fig. 5](../../test_results/nature_fig4_temporal_and_lincs_scaling.png)

**Fig. 5 | Temporal cascades, 18-day state-anchored extrapolation and LINCS scaling.** **a**, 0–24 h pathway cascade inferred on unseen MIX-Seq cell lines. **b**, Cross-species 18-day reprogramming extrapolation from D0–D12 to blind D12.5–D18, including persistence from D12. **c**, MSE scaling across 1,794 LINCS L1000 compounds with 359 unseen test drugs.

![Fig. 6](../../test_results/nature_fig5_clinical_and_safety.png)

**Fig. 6 | Clinical cohort translation and multi-organ safety.** **a**, Four-arm ablation across breast cancer treatment cohorts comparing intrinsic biomolecular, microenvironment, whole-transcriptome and fused predictors. **b**, Blind screening of 1,770 dual-drug combinations in 531 chemotherapy-resistant breast cancers against an eight-organ toxicity Pareto frontier. **c**, Unified GTEx v8 (54 organs) and EyeGEx target safety guard.

---

## Tables

**Table 1 | Cross-lineage out-of-distribution perturbation benchmark (seven lineages to blind unseen RPE1, *n* = 2,016 perturbations).** Mean *r*, Top-20 DEG *r* with bootstrap 95% confidence interval, Top-20 direction accuracy, global MSE, delta-weighted MSE (WMSE), perturbation discrimination score (PDS, median in parentheses) and win rate against the AIVC ensemble with two-sided paired Wilcoxon *P*. Arrows indicate the preferred direction.

| Model | Mean *r* | Top-20 DEG *r* [95% CI] | Top-20 DirAcc | Global MSE | WMSE (↓) | PDS (↑) | Win rate (*P*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AIVC ensemble** | 0.3965 | **0.5841** [0.5680, 0.5996] | **0.8133** | 0.011498 | 0.047649 | **0.6540** (0.7720) | reference |
| **AIVC v1.1** | **0.3973** | 0.5682 [0.5520, 0.5837] | 0.8110 | **0.011237** | **0.045925** | 0.6376 (0.7313) | — |
| Linear ridge | 0.3950 | 0.5209 [0.5026, 0.5369] | 0.7975 | 0.012365 | 0.053396 | 0.5312 (0.5337) | 63.8% (1.0 × 10⁻⁴³) |
| scFoundation-100M | 0.3354 | 0.4740 [0.4564, 0.4911] | 0.7654 | 0.011735 | 0.050038 | 0.5852 (0.6218) | 72.3% (< 1 × 10⁻⁵⁰) |
| scGPT | 0.3755 | 0.4503 [0.4312, 0.4676] | 0.7779 | 0.012463 | 0.054139 | 0.5013 (0.5015) | 72.9% (< 1 × 10⁻⁵⁰) |
| Train mean | 0.3762 | 0.4489 [0.4299, 0.4659] | 0.7773 | 0.012595 | 0.054806 | 0.5000 (random) | 73.2% (< 1 × 10⁻⁵⁰) |
| Geneformer-V2-104M | 0.3913 | 0.4463 [0.4270, 0.4630] | 0.7767 | 0.012285 | 0.053250 | 0.5031 (0.5017) | 72.6% (< 1 × 10⁻⁵⁰) |
| GEARS | 0.3515 | 0.4403 [0.4221, 0.4564] | 0.7709 | 0.012422 | 0.053823 | 0.5197 (0.5208) | 77.9% (< 1 × 10⁻⁵⁰) |
| scVI | 0.2685 | 0.4219 [0.4059, 0.4377] | 0.7585 | 0.012776 | 0.053603 | 0.5318 (0.5454) | 77.3% (< 1 × 10⁻⁵⁰) |
| Static MLP | 0.1186 | 0.1004 [0.0845, 0.1152] | 0.5466 | 0.014267 | 0.062743 | 0.5516 | 88.8% (< 1 × 10⁻⁵⁰) |
| No change | 0.0000 | 0.0000 | — | 0.014690 | 0.064577 | 0.5000 (random) | *P* = 2.9 × 10⁻²⁷⁵ |

**Table 2 | Unseen double-gene combinatorial epistasis benchmark (Norman2019, *n* = 27 unseen pairs).** Mean *r*, Top-20 DEG *r*, Top-20 direction accuracy, MSE and non-additive epistasis residual *r*. Rows labelled *equal-input* receive the identical *d*~A~ + *d*~B~ additive anchor.

| Model | Mean *r* | Top-20 DEG *r* | Top-20 DirAcc | MSE | Epistasis *r* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AIVC-Combo-Epistasis Ens5** | **0.9207** | **0.9642** | **0.9981** | **0.00119** | 0.6173 |
| Gain-scaled additive + ridge residual *(equal-input)* | 0.9099 | 0.9558 | 0.9944 | 0.00176 | **0.6479** |
| Additive × fitted gain (*s* = 0.829) *(equal-input)* | 0.8985 | 0.9461 | 0.9963 | 0.00196 | 0.5744 |
| Additive (*d*~A~ + *d*~B~) | 0.8985 | 0.9461 | 0.9963 | 0.00278 | 0.0000 |
| Linear ridge *(unanchored)* | 0.8004 | 0.8581 | 0.9519 | 0.00245 | 0.5361 |
| scGPT | 0.6764 | 0.8266 | 0.9278 | 0.00624 | 0.4501 |
| Static MLP | 0.7271 | 0.8237 | 0.9519 | 0.00347 | 0.5273 |
| scFoundation (100M) | 0.5308 | 0.7793 | 0.8963 | 0.01569 | 0.2796 |
| Geneformer-V2 (104M) | 0.6501 | 0.7553 | 0.9019 | 0.00473 | 0.4828 |
| Train mean | 0.5274 | 0.6208 | 0.8463 | 0.00709 | 0.4210 |
| GEARS | 0.4087 | 0.5653 | 0.8204 | 0.00824 | 0.3863 |

**Table 3 | Temporal benchmarks by evaluation subset and component ablations.** (a) Every model on every MIX-Seq subset, including the subsets on which baselines match or exceed AIVC. (b) Ablations, each computed within a single model instantiation; the struct-only row is a within-backbone contrast and is not comparable in absolute level to the fast–slow model.

| Subset (*n*) | Model | Mean *r* | Top-20 *r* | Top-20 DirAcc | MSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 6 h paired (36) | AIVC-ODE (fast–slow) | 0.5786 | 0.6305 | 0.9069 | 0.0238 |
| 6 h paired (36) | Joint grey-box variant | 0.4482 | 0.5399 | 0.8444 | 0.0372 |
| 6 h paired (36) | **Static MLP** | **0.6768** | **0.7970** | **0.9500** | **0.0207** |
| 6 h paired (36) | Static linear | 0.2599 | 0.5456 | 0.5986 | 0.0362 |
| 24 h paired (36) | AIVC open-loop rollout | 0.2443 | 0.3880 | 0.7139 | 0.0278 |
| 24 h paired (36) | AIVC Kalman-assimilated (*K* = 0.1) | 0.2425 | 0.3818 | 0.7153 | 0.0286 |
| 24 h paired (36) | 6 h persistence | 0.2035 | 0.2189 | 0.6069 | 0.0382 |
| 24 h paired (36) | **Static linear** | **0.4919** | **0.5649** | **0.8125** | **0.0126** |
| 24 h paired (36) | Static MLP | 0.0409 | 0.0877 | 0.5611 | 0.0295 |
| 24 h all 13 drugs (335) | AIVC-ODE (fast–slow) | 0.4860 | 0.6244 | 0.8351 | 0.0179 |
| 24 h all 13 drugs (335) | Joint grey-box variant | **0.5345** | **0.6472** | **0.8828** | **0.0133** |
| 24 h all 13 drugs (335) | Static MLP | 0.4463 | 0.5392 | 0.8175 | 0.0155 |
| 24 h all 13 drugs (335) | Static linear | 0.3590 | 0.4442 | 0.7254 | 0.0206 |

| Ablation | Comparison | Change | Interpretation |
| :--- | :--- | :---: | :--- |
| OOD CRISPRi (RPE1) | AIVC ensemble vs ridge | 0.5841 vs 0.5209 | Gated low-rank basis plus leave-one-lineage-out anchor closes the Delta Gap |
| Prior removed (RPE1) | zeros vs signed, same architecture | 0.5864 vs 0.5841 | Prior contributes nothing to transcriptome-level accuracy |
| Prior sign removed (RPE1) | unsigned vs signed | 0.5870 vs 0.5841 | Edge direction carries no detectable information |
| Prior removed, edge subset (RPE1, *n* = 613) | zeros vs signed | 0.6283 vs 0.6247 | No effect even where OmniPath out-edges exist |
| Prior on masked kinase (TCGA-OV) | with vs without OmniPath causal term | 0.1638 vs 0.1096 | Prior contributes where the readout is pathway propagation |
| Prior on masked kinase (CPTAC-LUAD) | with vs without OmniPath causal term | 0.0454 vs 0.0286 | Same direction, small-sample subset (*n* = 18 sites) |
| Cross-modal pretraining | Tahoe-pretrained vs matched single model | 0.5651 vs 0.5682 (−0.0031) | Chemical stress produces mild negative transfer |
| Kinase masking (TCGA-OV) | Masked vs unmasked | 0.1638 vs 0.2594 | 58–63% of unmasked signal retained |
| Overexpression (+Joung) | AIVC vs ridge | 0.5876 vs 0.4724 | Lineage-invariant branch prevents domain shift |
| Control-state encoder removed (RPE1) | same architecture, encoder zeroed | 0.4917 vs 0.5841 (−0.092) | Largest single component; weak tertile falls 0.322 → 0.213 |
| Teacher embeddings removed (RPE1) | co-expression + ESM-2 zeroed | 0.5495 vs 0.5841 (−0.035) | Second-largest component |
| Co-expression only removed (RPE1) | co-expression zeroed | 0.5754 vs 0.5841 (−0.009) | CI [−0.015, −0.002], separates from zero |
| ESM-2 only removed (RPE1) | ESM-2 zeroed | 0.5874 vs 0.5841 (+0.003) | No detectable contribution (CI includes zero) |
| All inputs zeroed (RPE1) | gated ridge backbone alone | 0.4451 vs 0.5841 (−0.139) | Upper bound of total input contribution |
| Post-hoc calibration (RPE1) | OOF-fitted scale/blend applied once | no arm improves without loss | Reported metrics are not a calibration artefact |
| Double-gene combo | AIVC vs gain-scaled ridge residual | MSE 0.00119 vs 0.00176 (−32.4%) | Nonlinear cross-talk beats linear residual |
| Zero-fill ablation | sci-Plex3 zero-fill vs clean, same architecture | 0.3644 vs 0.4860 (−0.1216) | Cross-platform zero-fill causes severe negative transfer |
| Same-modality pretraining | Tahoe vs scratch, struct-only backbone | 0.4727 → 0.4913 (+0.0186) | Within-backbone gain; absolute level below fast–slow model |
| 18-day reprogramming | Anchored ODE vs open-loop D0 | 0.8743 vs 0.7206 | Beats D0 rollout but matches D12 persistence (Δ = 0.0003) |

---

## Supplementary information

Supplementary Figs. 1–12, Supplementary Table 1 and the Reporting Summary are provided as a single PDF. Supplementary Fig. 1, ensemble uncertainty calibration; 2, locally gated five-omics translation; 3, causal topology and masked-kinase recovery; 4, L1-sparse learned importance; 5, full 80-pair target-discovery audit; 6, driver-versus-passenger attribution; 7, 54-organ safety atlas; 8, single-cell perturbation manifold; 9, phase-plane dynamics; 10, volcano and driver waterfall; 11–12, schematic mechanism diagrams (illustrative, not quantitative). Supplementary Table 1 lists accessions and release versions for every primary dataset.

---
