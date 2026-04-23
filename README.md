# Project 4: DDR Biomarker & Drug Response Pipeline

**Research question:** Which DDR mutations predict sensitivity to PARP and ATR inhibitors?

This is the fourth project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). My thesis identified ARID1A loss as a potential DDR vulnerability in GEA. This project directly tests that hypothesis computationally — integrating GDSC2 drug sensitivity data with DepMap mutation profiles to explore whether specific DDR mutations predict response to targeted therapies.

## At a Glance

| | |
|---|---|
| **Stack** | scikit-learn (LogReg · GradBoost) · SHAP · Mann-Whitney/BH FDR · Docker |
| **Data** | GDSC v17 (IC50 + mutation matrix via `gdsctools`), 988 cell lines, 5 PARP/DDR drugs |
| **POC headline** | Talazoparib HRD ρ=+0.121 (p=3e-4, n=912, survives Bonferroni); CV AUC 0.68–0.77 with tissue covariate; MSI-H preferentially sensitive to ATMi (p=0.022) + CHKi (p=0.019) |
| **Role** | DDR/synthetic-lethality framing from thesis ARID1A finding; implementation AI-assisted |
| **Portfolio** | Project 4 of 7 · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

## What It Does

Integrates pharmacogenomics and mutation data to explore DDR biomarkers:

1. **Data integration** — GDSC2 IC50 values joined with DepMap mutation profiles
2. **Feature engineering** — HRD scoring, MSI status classification, DDR pathway activity
3. **Model training** — Logistic regression and gradient boosting with stratified CV
4. **Biomarker analysis** — SHAP-based feature ranking, Mann-Whitney tests with BH FDR correction
5. **Visualization** — SHAP summary plots, clustermap of biomarker–drug associations

The pipeline connects ARID1A loss and MSI-H status to PARP/ATR inhibitor sensitivity signals — extending the thesis findings into a computational framework.

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

## Project Structure

```
project-4-ddr-biomarker-pipeline/
├── README.md
├── Dockerfile
├── environment.yml
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── config.py
├── src/
│   ├── __init__.py
│   ├── biomarker_analysis.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── utils.py
├── scripts/
│   ├── run_pipeline.py
│   ├── run_biomarker_analysis.py
│   ├── run_evaluation.py
│   └── poc/
│       └── run_poc.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── data/
└── results/
    └── poc/
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-4-ddr-biomarker-pipeline.git
cd project-4-ddr-biomarker-pipeline

pip install -r requirements.txt
python scripts/run_pipeline.py
```

## Proof of Concept (v2)

An end-to-end DDR biomarker analysis on real GDSC cell-line data, testing (a) whether a mutation-based HRD score predicts PARP/DDR inhibitor sensitivity and (b) whether MSI-H cell lines are preferentially sensitive to ATM/CHK inhibitors.

**Dataset:** GDSC v17 (IC50 values + binary mutation matrix) bundled with the `gdsctools` PyPI package (CancerRxGene / Sanger Institute). The canonical `cog.sanger.ac.uk` release-8.5 hosts are unreachable from the reproducibility sandbox, so v17 is the accessible snapshot.

**Feature block:**
- 988 cell lines (all of v17)
- 11 DDR genes retained in v17 mutation matrix (of 26 queried)
- `HRD_narrow` = BRCA1 + BRCA2 + ATM mutation count (0–3)
- `HRD_broad`  = sum of all 11 DDR-gene mutations
- 60 MSI-H cell lines

**Drugs analysed:** Olaparib, Rucaparib, Talazoparib (PARPi), KU-55933 (ATM inhibitor), AZD7762 (CHK1/2 inhibitor). Originally-planned ATR inhibitors (VE-821, AZD6738) are absent from v17 — KU-55933 and AZD7762 stand in as DDR-pathway controls.

**Reproduce:**
```bash
pip install pandas numpy scipy scikit-learn shap matplotlib gdsctools
python scripts/poc/run_poc.py
```

### Per-drug Spearman (HRD_broad vs ln IC50)

| Drug | N | Spearman ρ | p-value |
|---|---|---|---|
| Olaparib | 847 | +0.068 | 0.048 |
| Rucaparib | 918 | +0.083 | 0.012 |
| **Talazoparib** | **912** | **+0.121** | **0.00026** ← survives Bonferroni ×5 |
| KU-55933 (ATMi) | 845 | +0.002 | 0.953 |
| AZD7762 (CHKi) | 846 | +0.091 | 0.008 |

### Per-drug 5-fold CV ROC-AUC (logistic regression, balanced classes)

| Drug | Feature set | N | CV AUC |
|---|---|---|---|
| Olaparib | DDR + MSI_FACTOR | 424 | 0.603 ± 0.051 |
| Rucaparib | DDR + MSI_FACTOR | 460 | 0.522 ± 0.046 |
| Talazoparib | DDR + MSI_FACTOR | 456 | 0.583 ± 0.052 |
| KU-55933 | DDR + MSI_FACTOR | 424 | 0.587 ± 0.054 |
| AZD7762 | DDR + MSI_FACTOR | 424 | 0.594 ± 0.049 |
| Olaparib | **DDR + MSI + Tissue** | 424 | **0.736 ± 0.050** |
| Rucaparib | DDR + MSI + Tissue | 460 | 0.681 ± 0.046 |
| Talazoparib | DDR + MSI + Tissue | 456 | 0.713 ± 0.030 |
| KU-55933 | DDR + MSI + Tissue | 424 | 0.750 ± 0.025 |
| AZD7762 | DDR + MSI + Tissue | 424 | **0.772 ± 0.036** |

Tissue context adds **10–25 AUC points per drug**, quantifying how much signal is lineage-driven rather than DDR-specific.

### MSI-H vs MSS drug sensitivity (Mann-Whitney on ln IC50)

| Drug | N MSI-H | N MSS | MW p-value |
|---|---|---|---|
| Olaparib | 54 | 793 | 0.083 |
| Rucaparib | 56 | 862 | 0.493 |
| Talazoparib | 58 | 854 | 0.039 |
| **KU-55933 (ATMi)** | 56 | 789 | **0.022** ← MSI-H sensitive |
| **AZD7762 (CHKi)** | 54 | 792 | **0.019** ← MSI-H sensitive |

### Headline numbers

- Best HRD correlation: **Talazoparib ρ=+0.121, p=3e-4, n=912**
- Best CV AUC (DDR + MSI + Tissue): **AZD7762 AUC 0.772 ± 0.036**
- All 5 drugs improve substantially with tissue context (0.52–0.60 → 0.68–0.77)
- MSI-H subgroup preferentially sensitive to ATM (p=0.022) and CHK (p=0.019) inhibitors — matches synthetic-lethality literature

### Comparison to v1 (legacy, 4 HR genes only)

| Metric | v1 (BRCA1+BRCA2+ATM+CHEK2) | v2 (11 DDR + MSI + tissue) |
|---|---|---|
| Olaparib ρ | +0.076, p=0.026 | +0.068, p=0.048 |
| Olaparib CV AUC | 0.529 ± 0.022 | **0.736 ± 0.050** |
| Best CV AUC | — | **0.772 ± 0.036** (AZD7762) |

### Honest assessment

- Published Spearman ρ for mutation-only HRD vs PARPi ln IC50 in cell-line panels is **−0.15 to −0.30** (weak even with HRD-scar scores).
- Our strongest correlation (Talazoparib ρ=+0.121) is modest but **survives Bonferroni correction** across 5 drugs (α=0.01).
- The **positive direction** (higher mutation burden → higher IC50 → more resistant) reflects the confounding effect of TP53 and general cancer-driver burden: cell lines with more mutations tend to be more resistant to most drugs, independent of DDR mechanism.
- The MSI-H subgroup is **preferentially sensitive to ATM and CHK inhibitors** — an independent signal that matches the synthetic-lethality literature.
- Tissue context adds 10–25 AUC points per drug, quantifying how much of the signal is lineage-driven rather than DDR-specific.
- Negative findings are scientifically informative.

### Limits

- Mutation-only HRD is a crude proxy. Published HRD scar scores (HRD-LOH, LST, TAI, Myriad HRD) perform noticeably better.
- 15 of 26 intended DDR genes are unavailable in v17 (PALB2, RAD51 family, BARD1, BRIP1, NBN, MRE11, RAD50, FANC family).
- VE-821 and AZD6738 (ATRi) are in GDSC 8.x but not v17, so ATR sensitivity is not assessed directly.
- Cell-line analysis only — no TCGA validation cohort.
- Tissue as a covariate may be doing most of the work in the 0.68–0.77 AUC range; DDR/MSI contribute less than tissue alone in several drugs.

## My Role

I led the DDR synthetic lethality framing directly from my thesis findings on ARID1A. I directed feature construction around HRD biology and reviewed all outputs for biological coherence. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 4 of 7**. It narrows Project 3's broad pharmacogenomics approach to the specific DDR biology from my thesis. The biomarker features developed here (MSI status, DDR burden, tissue context) feed directly into the survival model in Project 6. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
