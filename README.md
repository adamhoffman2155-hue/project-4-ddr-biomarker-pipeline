# Project 4: DDR Biomarker & Drug Response Pipeline

**Research question:** Which DDR mutations predict sensitivity to PARP and ATR inhibitors?

This is the fourth project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). My thesis identified ARID1A loss as a potential DDR vulnerability in GEA. This project directly tests that hypothesis computationally — integrating GDSC2 drug sensitivity data with DepMap mutation profiles to explore whether specific DDR mutations predict response to targeted therapies.

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

The pipeline scores mutations across a curated DDR gene set:
BRCA1, BRCA2, ATM, ATR, PALB2, RAD51, MLH1, MSH2, MSH6, POLE, ARID1A, CDK12, CHEK2

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

# Choose one environment
docker build -t ddr-biomarker . && docker run -it -v $(pwd):/workspace ddr-biomarker bash
#   or
conda env create -f environment.yml && conda activate ddr-biomarker-pipeline
#   or
pip install -r requirements.txt

# Full pipeline
python scripts/run_pipeline.py

# Individual stages
python scripts/run_biomarker_analysis.py
python scripts/run_evaluation.py
```

## Proof of Concept

An end-to-end DDR biomarker analysis on real GDSC cell-line data, testing whether a mutation-based HRD score predicts sensitivity to PARP/DDR inhibitors.

**Dataset:** GDSC v17 (IC50 values + binary mutation matrix) bundled with the `gdsctools` PyPI package (CancerRxGene / Sanger), plus GDSC release 8.5 drug metadata mirrored on GitHub. The canonical `cog.sanger.ac.uk` release 8.5 hosts are unreachable from the reproducibility sandbox, so v17 is the accessible snapshot.

**HRD score:** count of mutations across an HR gene panel. Intended panel was 19 genes; the v17 genomic_features matrix pre-filters to frequently-mutated cancer genes, retaining only **4 of 19** (BRCA1, BRCA2, ATM, CHEK2). This truncation is expected to attenuate the correlation vs published HRD-PARPi benchmarks.

**Drugs analysed:** Olaparib, Rucaparib, Talazoparib (PARPi), KU-55933 (ATM inhibitor), AZD7762 (CHK1/2 inhibitor). The originally-planned ATR inhibitors (VE-821, AZD6738) are absent from v17 — KU-55933 and AZD7762 are DDR-pathway stand-ins.

**Reproduce:**
```bash
pip install pandas numpy scipy scikit-learn shap matplotlib gdsctools
python scripts/poc/run_poc.py
```

**Spearman correlations (HRD score vs ln IC50, actual values):**

| Drug | N | Spearman ρ | p-value |
|---|---|---|---|
| Olaparib | 847 | **+0.076** | 0.026 |
| Rucaparib | 918 | +0.064 | 0.052 |
| Talazoparib | 912 | +0.044 | 0.189 |
| KU-55933 (ATMi) | 845 | +0.037 | 0.278 |
| AZD7762 (CHKi) | 846 | +0.030 | 0.382 |

**Per-drug 5-fold CV ROC-AUC (LogReg on 4 HR genes):**

| Drug | N | N sensitive | CV AUC |
|---|---|---|---|
| Olaparib | 847 | 212 | 0.529 ± 0.022 |
| Rucaparib | 918 | 230 | 0.510 ± 0.010 |
| KU-55933 | 845 | 212 | 0.515 ± 0.011 |
| AZD7762 | 846 | 212 | 0.498 ± 0.006 |
| Talazoparib | 912 | 228 | 0.483 ± 0.006 |

**HRD score distribution (0–4 possible):**

| HRD score | N cell lines |
|---|---|
| 0 | 912 |
| 1 | 66 |
| 2 | 9 |
| 3 | 1 |

**Top SHAP features for Olaparib (4 HR genes only):**

| Gene | mean \|SHAP\| | direction |
|---|---|---|
| BRCA1 | 0.120 | protective (sensitive) |
| ATM | 0.066 | neutral |
| BRCA2 | 0.040 | mildly resistant direction |
| CHEK2 | 0.021 | protective (sensitive) |

**Honest assessment:**
- Correlations are **very weak and in the wrong direction** (+0.04 to +0.08 for PARPi; expected is -0.15 to -0.3 per published cell-line benchmarks).
- CV AUCs are **~0.5–0.53** — essentially random for all drugs.
- The root cause is documented: v17 genomic_features only retains 4 of 19 intended HR-panel genes, so the HRD score has a max of 4 and is 0 for 92% of cell lines. With so little signal to work with, a weak/null result is expected.
- **This is a useful negative result:** mutation-only HRD scoring on a truncated gene panel does not recover PARPi sensitivity signal. To improve, a full cell-line WES-derived HRD-scar score (HRD-LOH, LST, TAI) would be needed.
- BRCA1 still has the highest SHAP importance for Olaparib, pointing in the sensitivity direction, which is the one piece of biology the model does recover.

**Limits:**
- Mutation-only HRD is a crude proxy. Published HRD scar scores (HRD-LOH, LST, TAI, Myriad HRD) perform noticeably better.
- 15 of 19 intended HR genes are unavailable in v17 (PALB2, RAD51 family, BARD1, BRIP1, NBN, MRE11, RAD50, FANC family).
- VE-821 and AZD6738 are in GDSC 8.x but not v17, so ATRi sensitivity is not assessed directly.
- Cell-line analysis only — no TCGA validation cohort.

## My Role

I led the DDR synthetic lethality framing directly from my thesis findings on ARID1A. I directed feature construction around HRD biology and reviewed all outputs for biological coherence. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 4 of 7**. It narrows Project 3's broad pharmacogenomics approach to the specific DDR biology from my thesis. The biomarker features developed here (MSI status, DDR burden) feed directly into the survival model in Project 6. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
