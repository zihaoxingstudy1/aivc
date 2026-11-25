# Submission checklist — what remains before the preprint goes live

Everything that can be fixed without new information has been fixed. This page lists what only the author can supply, where it goes, and what format it takes. Fill these once, then both bioRxiv and Nature Communications can be submitted without a second revision.

---

## 1. Author block (manuscript lines 3–7, 340)

| Item | Current placeholder | Replace with |
| :--- | :--- | :--- |
| Author names, in order | `[First Author]¹*, [Co-Authors]¹` | Full names, superscript affiliations, asterisk on the corresponding author |
| Affiliation | `[Department / Institution, City, Postal Code, Country]` | One line per institution, numbered if more than one |
| Correspondence email | `[corresponding.author@institution.edu]` | The address you will actually monitor during review |
| ORCID (optional but recommended) | — | Add after the name, e.g. `0000-0002-XXXX-XXXX` |

The same email must be repeated in **Materials and correspondence** (line 340) and in the cover letter signature.

**Do this before the preprint.** A preprint is a permanent record; changing the author list afterwards requires a new version and leaves a visible correction trail.

---

## 2. Repository identifiers

| Item | Current placeholder | Where it appears |
| :--- | :--- | :--- |
| GitHub URL | `[repository URL to be supplied]` | Code availability |
| Zenodo DOI (code) | `[Zenodo DOI to be supplied]` | Code availability |
| Repository DOI (data/derived) | `[repository DOI to be supplied]` | Data availability |

Order of operations: create the GitHub repository → tag `v1.1.0` → enable the Zenodo integration → publish the release → Zenodo mints the DOI → paste both identifiers here.

---

## 3. Preprint disclosure (cover letter)

The cover letter currently carries a two-branch sentence:

> [If posted before submission: this work has been posted as a preprint at bioRxiv, DOI 10.1101/XXXXXXX, released under CC BY 4.0. / If not posted: this work has not been posted as a preprint.]

Keep exactly one branch and delete the other. Nature Portfolio requires the preprint DOI and licence to be disclosed at submission, so leaving both branches in would read as an unresolved placeholder.

---

## 4. Supplementary Table 1 — accessions

Two rows are confirmed and already filled:

- Norman 2019 — **GSE133344** (from the Science paper's Data availability statement)
- sci-Plex3 (Srivatsan 2020) — **GSE139944** (verified against the GEO record, PMID 31806696)

Fourteen rows remain. Each names the reference whose Data availability section carries the accession; copy from there, not from a secondary source:

| Dataset | Reference in the manuscript |
| :--- | :--- |
| Replogle Perturb-seq | ref. 6 |
| JoungZhang2023 | ref. 11 |
| MIX-Seq | ref. 12 |
| SchiebingerLander2019 | ref. 14 |
| Tahoe-100M | resource paper (not yet cited; add citation or cite the portal) |
| TCGA | GDC — record the freeze (e.g. `TCGA 2024-XX-XX`) |
| CPTAC | CPTAC Portal — record the release |
| CCLE / DepMap | record the quarter, e.g. `DepMap 24Q2` |
| GDSC1/2 | record the download date |
| PRISM 20Q2 | DepMap — fixed release name |
| DrugComb | record the version date |
| OmniPath | record the download date |
| BioGRID | record the release, e.g. `4.4.2xx` |
| MSigDB / KEGG / Reactome / GO-BP | record the release of each |

Record the release **actually used**, not the current one. Reviewers check this against the numbers in the paper.

Two datasets were checked during this pass and deliberately left blank because the candidate accessions did not match the papers: Replogle 2022 and Schiebinger 2019. Their GEO series do not appear under the obvious search terms, so take the accession directly from each paper's own Data availability statement.

---

## 5. Declarations already settled

- **Funding** — recorded as self-funded, with no external grant and no funder role. No action needed unless a grant should be added.
- **Competing interests** — recorded as author-held IP with MIT code / CC BY-NC-SA weights. Confirm with your technology-transfer office and add any pending patent numbers.
- **Ethics** — recorded as reanalysis of public de-identified data; no approval required.

---

## 6. Pre-release checks (5 minutes, run in this order)

1. Replace every placeholder above; then search the manuscript for `[` and confirm only matrix indices remain.
2. Confirm the author block, email and affiliation are identical in manuscript, cover letter and submission system.
3. Confirm the preprint DOI in the cover letter matches the record once bioRxiv assigns it.
4. Confirm `LICENSE` says MIT and `pyproject.toml` agrees — a mismatch here is the first thing an open-source reviewer notices.
5. Commit, tag `v1.1.0`, publish the Zenodo release, then post the preprint.