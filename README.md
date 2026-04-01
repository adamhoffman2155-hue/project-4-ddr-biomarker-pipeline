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
├── config/
│   └── config.py                  # Pipeline config, gene panels, hyperparameters
├── src/
│   ├── data_loader.py             # GDSC2 + DepMap data loading
│   ├── feature_engineering.py     # HRD scoring, MSI, DDR pathway features
│   ├── models.py                  # LogReg, GBM with stratified CV
│   ├── biomarker_analysis.py      # SHAP, statistical tests, clustermap
│   ├── evaluation.py              # ROC/PR curves, model comparison
│   └── utils.py                   # Helpers
├── scripts/
│   ├── run_pipeline.py            # Full pipeline CLI
│   ├── run_biomarker_analysis.py  # Biomarker-only analysis
│   └── run_evaluation.py          # Evaluation-only
├── tests/
│   └── test_pipeline.py
├── data/
├── results/
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-4-ddr-biomarker-pipeline.git
cd project-4-ddr-biomarker-pipeline

pip install -r requirements.txt
python scripts/run_pipeline.py
```

## My Role

I led the DDR synthetic lethality framing directly from my thesis findings on ARID1A. I directed feature construction around HRD biology and reviewed all outputs for biological coherence. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 4 of 7**. It narrows Project 3's broad pharmacogenomics approach to the specific DDR biology from my thesis. The biomarker features developed here (MSI status, DDR burden) feed directly into the survival model in Project 6. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
