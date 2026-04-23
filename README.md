# Project 4: DDR Biomarker & Drug Response Pipeline

> **A pipeline that tests whether specific DNA-repair defects predict response to a family of targeted cancer drugs (PARP and ATR inhibitors).**

## The short version

**What this project does.** Across 988 cancer cell lines and 5 DNA-damage-response drugs, tests two biological hypotheses: (1) more DNA-repair mutations → more sensitive to these drugs, and (2) microsatellite-unstable (MSI-high) cell lines are specifically sensitive to ATM and CHK inhibitors.

**The question behind it.** My M.Sc. thesis identified a gene (ARID1A) as a potential vulnerability in stomach/esophageal cancer. If that's right, tumors with this defect should be unusually sensitive to drugs that exploit broken DNA-repair. This project tests that hypothesis quantitatively across a large cell-line panel.

**What the proof-of-concept shows.**
- The strongest correlation is for **Talazoparib** (a PARP inhibitor): more DNA-repair mutations → more resistant, ρ = +0.12, p = 0.0003 (survives correction for testing 5 drugs).
- Adding tumor **tissue type** as a feature pushes classification accuracy from ~55% to **68–77%** per drug — **most of the signal is tissue-driven**, not DDR-driven.
- **Concrete biological finding:** MSI-high cell lines are preferentially sensitive to ATM inhibitors (p = 0.022) and CHK inhibitors (p = 0.019) — exactly matching the synthetic-lethality literature.

**Why it matters.** The numbers bound honestly how far mutation-only biomarkers can take us, and they point to where the bigger signals actually live (tissue context, MSI status, HRD scar scores). The MSI → ATMi/CHKi result is a genuine, independently verifiable finding that reproduces published biology.

---

_The rest of this README is technical detail for bioinformaticians, recruiters doing a deep review, or anyone reproducing the work._

## At a Glance

| | |
|---|---|
| **Stack** | scikit-learn (LogReg · GradBoost) · SHAP · Mann-Whitney/BH FDR · Docker |
| **Data** | GDSC v17 (IC50 + mutation matrix via `gdsctools`), 988 cell lines, 5 PARP/DDR drugs |
| **POC headline** | Talazoparib HRD ρ=+0.121 (p=3e-4, n=912, survives Bonferroni); CV AUC 0.68–0.77 with tissue covariate; MSI-H preferentially sensitive to ATMi (p=0.022) + CHKi (p=0.019) |
| **Status** | **Runnable POC** with committed per-drug CV AUCs, Spearman table, MSI-H test, SHAP |
| **Role** | DDR/synthetic-lethality framing from thesis ARID1A finding; implementation AI-assisted |
| **Portfolio** | Project 4 of 7 · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

## What It Does

Integrates pharmacogenomics and mutation data to explore DDR biomarkers:

1. **Data integration** — GDSC2 IC50 values joined with DepMap mutation profiles
2. **Feature engineering** — HRD scoring, MSI status classification, DDR pathway activity
3. **Model training** — Logistic regression and gradient boosting with stratified CV
4. **Biomarker analysis** — SHAP-based feature ranking, Mann-Whitney tests with BH FDR correction
5. **Visualization** — SHAP summary plots, clustermap of biomarker–drug associations

## Methods & Tools

| Category | Tools |
|----------|-------|
| Data Sources | GDSC2, DepMap |
| ML Models | Logistic Regression, Gradient Boosting (scikit-learn) |
| Biomarker Analysis | SHAP, Mann-Whitney U, Benjamini-Hochberg FDR |
| Feature Engineering | HRD scoring, DDR gene panel, MSI classification |
| Visualization | matplotlib, seaborn, clustermap |
| Environment | Docker, Conda |

## DDR Gene Panel

The pipeline targets a 26-gene DDR panel including BRCA1, BRCA2, ATM, ATR, PALB2, RAD51, MLH1, MSH2, MSH6, POLE, ARID1A, CDK12, CHEK2, TP53, and others. The GDSC v17 mutation matrix retains **11 of 26** (BRCA1, BRCA2, ATM, ATR, CHEK2, MLH1, MLH3, TP53, CDKN2A, RB1, PTEN) — the rest are filtered by v17's frequent-cancer-gene heuristic.

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-4-ddr-biomarker-pipeline.git
cd project-4-ddr-biomarker-pipeline

pip install -r requirements.txt
python scripts/poc/run_poc.py
```

## Proof of Concept (v2)

An end-to-end DDR biomarker analysis on real GDSC cell-line data, testing (a) whether a mutation-based HRD score predicts PARP/DDR inhibitor sensitivity and (b) whether MSI-H cell lines are preferentially sensitive to ATM/CHK inhibitors.

**Dataset:** GDSC v17 (IC50 values + binary mutation matrix) bundled with the `gdsctools` PyPI package. The canonical `cog.sanger.ac.uk` release-8.5 hosts are unreachable from the reproducibility sandbox, so v17 is the accessible snapshot.

**Feature block:** 988 cell lines, 11 DDR genes retained, HRD_narrow = BRCA1+BRCA2+ATM count, HRD_broad = all 11 DDR-gene count, 60 MSI-H cell lines.

**Drugs analysed:** Olaparib, Rucaparib, Talazoparib (PARPi), KU-55933 (ATM inhibitor), AZD7762 (CHK1/2 inhibitor).

### Per-drug Spearman (HRD_broad vs ln IC50)

| Drug | N | Spearman ρ | p-value |
|---|---|---|---|
| Olaparib | 847 | +0.068 | 0.048 |
| Rucaparib | 918 | +0.083 | 0.012 |
| **Talazoparib** | **912** | **+0.121** | **0.00026** ← survives Bonferroni ×5 |
| KU-55933 (ATMi) | 845 | +0.002 | 0.953 |
| AZD7762 (CHKi) | 846 | +0.091 | 0.008 |

### Per-drug 5-fold CV ROC-AUC

| Drug | Feature set | CV AUC |
|---|---|---|
| Olaparib | DDR + MSI | 0.603 ± 0.051 |
| Rucaparib | DDR + MSI | 0.522 ± 0.046 |
| Talazoparib | DDR + MSI | 0.583 ± 0.052 |
| KU-55933 | DDR + MSI | 0.587 ± 0.054 |
| AZD7762 | DDR + MSI | 0.594 ± 0.049 |
| Olaparib | **DDR + MSI + Tissue** | **0.736 ± 0.050** |
| Rucaparib | DDR + MSI + Tissue | 0.681 ± 0.046 |
| Talazoparib | DDR + MSI + Tissue | 0.713 ± 0.030 |
| KU-55933 | DDR + MSI + Tissue | 0.750 ± 0.025 |
| AZD7762 | DDR + MSI + Tissue | **0.772 ± 0.036** |

Tissue context adds **10–25 AUC points per drug**.

### MSI-H vs MSS drug sensitivity (Mann-Whitney on ln IC50)

| Drug | N MSI-H | N MSS | MW p-value |
|---|---|---|---|
| Talazoparib | 58 | 854 | 0.039 |
| **KU-55933 (ATMi)** | 56 | 789 | **0.022** ← MSI-H sensitive |
| **AZD7762 (CHKi)** | 54 | 792 | **0.019** ← MSI-H sensitive |

### Headline numbers

- Best HRD correlation: **Talazoparib ρ=+0.121, p=3e-4, n=912**
- Best CV AUC (DDR + MSI + Tissue): **AZD7762 AUC 0.772 ± 0.036**
- MSI-H subgroup preferentially sensitive to ATM (p=0.022) and CHK (p=0.019) inhibitors

### Honest assessment

- Published Spearman ρ for mutation-only HRD vs PARPi is **−0.15 to −0.30** — ours is modestly positive but survives Bonferroni.
- The positive-direction ρ reflects confounding by TP53 / cancer-driver burden: cell lines with more mutations tend to be more drug-resistant in general.
- MSI-H → ATMi/CHKi sensitivity is an independent, published-literature-matching signal.
- Tissue as a covariate does most of the work in the 0.68–0.77 AUC range.

### Limits

- Mutation-only HRD is a crude proxy vs. HRD scar scores (HRD-LOH, LST, TAI).
- 15 of 26 intended DDR genes are unavailable in v17.
- VE-821 and AZD6738 (ATRi) are in GDSC 8.x but not v17.
- Cell-line analysis only — no TCGA validation cohort.

## My Role

I led the DDR synthetic lethality framing directly from my thesis findings on ARID1A. I directed feature construction around HRD biology and reviewed all outputs for biological coherence. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 4 of 7**. It narrows Project 3's broad pharmacogenomics approach to the specific DDR biology from my thesis. The biomarker features developed here (MSI status, DDR burden, tissue context) feed directly into the survival model in Project 6. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
