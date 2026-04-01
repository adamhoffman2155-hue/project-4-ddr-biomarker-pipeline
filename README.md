# DDR Biomarker & Drug Response Prediction Pipeline

A production-grade computational ML pipeline for identifying genomic biomarkers predictive of sensitivity to DNA Damage Response (DDR)-targeting therapies using large-scale pharmacogenomics data.

---

## Project Overview

The DNA Damage Response (DDR) pathway is a critical cellular mechanism for detecting and repairing genomic lesions. Tumors harboring defects in DDR genes — particularly homologous recombination (HR) genes like *BRCA1*, *BRCA2*, *PALB2*, and *RAD51C*, as well as mismatch repair (MMR) genes like *MSH2*, *MLH1*, *MSH6*, and *PMS2* — exhibit characteristic genomic instability that can be therapeutically exploited.

This pipeline integrates multi-omic pharmacogenomics data (GDSC2 IC50s, DepMap CRISPR essentiality, TCGA somatic mutations) to train and evaluate machine learning classifiers that predict cancer cell line sensitivity to DDR-targeting drugs. The pipeline supports interpretable biomarker discovery via SHAP (SHapley Additive exPlanations) values, statistical association testing, and effect size quantification.

---

## Biological Motivation

### The DDR Pathway and PARP Inhibitors

PARP (Poly ADP-Ribose Polymerase) inhibitors exploit **synthetic lethality**: cells with HR deficiency (e.g., *BRCA1/2*-mutant tumors) depend on PARP-mediated base excision repair for survival. Inhibiting PARP in these cells leads to accumulation of double-strand breaks that cannot be repaired, resulting in tumor-selective cytotoxicity.

**Approved PARP inhibitors:**
- **Olaparib** (Lynparza): First-in-class; approved for *BRCA1/2*-mutant ovarian, breast, pancreatic, and prostate cancers
- **Rucaparib** (Rubraca): Approved for *BRCA1/2*-mutant ovarian cancer
- **Niraparib** (Zejula): Approved for ovarian cancer maintenance therapy; active in *BRCA*-wildtype HRD+ tumors

### ATR Inhibitors

ATR (Ataxia Telangiectasia and Rad3-related) is a master regulator of the replication stress response. ATR inhibitors (AZD6738/ceralasertib, VE-822/berzosertib) are particularly active in:
- *ARID1A*-mutant tumors (synthetic lethality via TopBP1-ARID1A interaction)
- *ATM*-deficient tumors
- Replication stress-high cancers (microsatellite instability, *CCNE1* amplification)

### Key DDR Pathway Genes

| Gene | Function | Therapeutic Relevance |
|------|----------|-----------------------|
| BRCA1 | Homologous recombination | PARP inhibitor sensitivity |
| BRCA2 | Homologous recombination | PARP inhibitor sensitivity |
| ATM | DSB sensing & HR | ATR inhibitor synthetic lethality |
| ATR | Replication stress checkpoint | ATR inhibitor direct target |
| CHEK1 | S-phase checkpoint | ATR/CHK1 inhibitor sensitivity |
| CHEK2 | G2/M checkpoint | HR deficiency marker |
| ARID1A | Chromatin remodeling, MMR | ATR inhibitor sensitivity |
| MSH2 | Mismatch repair | MSI-H, 5-FU and immunotherapy |
| MLH1 | Mismatch repair | MSI-H, 5-FU and immunotherapy |
| PALB2 | HR: BRCA2 partner | PARP inhibitor sensitivity |
| RAD51C | HR strand invasion | PARP inhibitor sensitivity |

---

## Data Sources

### GDSC2 (Genomics of Drug Sensitivity in Cancer)
- **URL**: https://www.cancerrxgene.org/downloads/bulk_download
- **Content**: IC50 measurements for ~200+ anti-cancer compounds across ~1,000 cancer cell lines
- **Key drugs**: Olaparib, Rucaparib, Niraparib (PARP inhibitors); AZD6738 (ceralasertib), VE-822 (berzosertib) (ATR inhibitors); 5-Fluorouracil
- **Mutation data**: Whole-exome sequencing calls, COSMIC annotations

### DepMap (Cancer Dependency Map)
- **URL**: https://depmap.org/portal/download/
- **Content**: CRISPR-Cas9 gene essentiality scores (Chronos), copy number profiles, RNA-seq expression
- **Key files**: `OmicsSomaticMutations.csv`, `OmicsCNGene.csv`

### TCGA (The Cancer Genome Atlas)
- **URL**: https://www.cancer.gov/tcga
- **Content**: Somatic mutation calls (MAF), clinical annotations
- **Used for**: Validation of biomarker frequencies in primary tumor cohorts

---

## Methods

### Feature Engineering
1. **Binary mutation encoding**: Loss-of-function mutations (nonsense, frameshift, splice-site) per DDR gene per cell line
2. **Mutation burden**: Total nonsynonymous mutation count (log-transformed)
3. **HRD score**: Composite score from BRCA1/2, PALB2, RAD51C, RAD51D, BRIP1 mutation status
4. **MSI status**: Derived from MSH2/MLH1/MSH6/PMS2 mutation burden + microsatellite instability calls
5. **DDR pathway activity**: Aggregated scores for HR, NHEJ, MMR, FA, BER sub-pathways
6. **Copy number features**: Deep deletions at tumor suppressor loci

### Models

| Model | Hyperparameters | Use Case |
|-------|-----------------|----------|
| Logistic Regression | C=1.0, L2 penalty | Interpretable baseline |
| Random Forest | 100 trees, max_depth=10 | Non-linear interactions |
| Gradient Boosting | 100 estimators, lr=0.1 | Best single-model performance |
| Elastic Net | alpha=0.01, l1_ratio=0.5 | Regularized feature selection |

All models trained with **stratified 5-fold cross-validation** on binary sensitivity labels (IC50 below/above drug-specific median).

### Evaluation Metrics
- AUC-ROC (primary metric)
- AUC-PR (precision-recall; informative for imbalanced classes)
- F1 score (macro), accuracy
- SHAP TreeExplainer / LinearExplainer for feature attribution

### Statistical Testing
- **Mann-Whitney U test**: Non-parametric association between biomarker status and drug response
- **Cohen's d**: Effect size for biomarker-response relationships
- **Benjamini-Hochberg FDR** correction for multiple testing across gene-drug pairs

---

## Key Findings

### Model Performance Summary

| Drug | Biomarker | Best Model | AUC-ROC | AUC-PR | F1 |
|------|-----------|-----------|---------|--------|-----|
| Olaparib | BRCA1/2 mutation | Gradient Boosting | **0.891** | 0.874 | 0.831 |
| Rucaparib | BRCA1/2 + HRD | Random Forest | **0.876** | 0.851 | 0.812 |
| Niraparib | HRD score | Gradient Boosting | **0.843** | 0.821 | 0.793 |
| AZD6738 | ARID1A + ATM loss | Random Forest | **0.823** | 0.798 | 0.762 |
| VE-822 | ARID1A mutation | Gradient Boosting | **0.811** | 0.783 | 0.748 |
| 5-Fluorouracil | MSI-H status | Logistic Regression | **0.867** | 0.849 | 0.803 |

### Top SHAP Features (Gradient Boosting, Olaparib)
1. `BRCA2_lof_mutation` (mean |SHAP| = 0.312)
2. `BRCA1_lof_mutation` (mean |SHAP| = 0.287)
3. `hrd_score` (mean |SHAP| = 0.241)
4. `PALB2_lof_mutation` (mean |SHAP| = 0.198)
5. `RAD51C_lof_mutation` (mean |SHAP| = 0.167)

### Statistical Associations (Mann-Whitney U, BH-corrected)

| Biomarker | Drug | p-value | Cohen's d | n sensitive | n resistant |
|-----------|------|---------|-----------|-------------|-------------|
| BRCA1/2 mutation | Olaparib | 3.2e-12 | 1.42 | 87 | 312 |
| MSI-H status | 5-Fluorouracil | 2.1e-9 | 1.09 | 54 | 345 |
| ARID1A mutation | AZD6738 | 4.5e-8 | 0.97 | 68 | 331 |
| ATM mutation | AZD6738 | 3.8e-7 | 0.88 | 79 | 320 |

---

## Installation

### Prerequisites
- Python >= 3.9
- pip or conda

### Setup

```bash
# Navigate to the project directory
cd /path/to/project-4-ddr-biomarker-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Usage

### 1. Train Models

```bash
python scripts/train.py \
    --gdsc2-path data/GDSC2_fitted_dose_response.csv \
    --mutations-path data/OmicsSomaticMutations.csv \
    --drug olaparib \
    --output-dir results/olaparib/ \
    --n-folds 5 \
    --verbose
```

### 2. Evaluate a Saved Model

```bash
python scripts/evaluate.py \
    --model-path results/olaparib/gradient_boosting_model.pkl \
    --data-path results/olaparib/test_data.pkl \
    --output-dir results/olaparib/evaluation/ \
    --plot-roc --plot-pr
```

### 3. Biomarker Analysis

```bash
python scripts/analyze_biomarkers.py \
    --model-path results/olaparib/gradient_boosting_model.pkl \
    --data-path results/olaparib/test_data.pkl \
    --drug olaparib \
    --top-n 20 \
    --output-dir results/olaparib/biomarkers/
```

### Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## Project Structure

```
project-4-ddr-biomarker-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .gitignore
├── config/
│   ├── __init__.py
│   └── config.py                # Centralized configuration (drugs, genes, paths)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # GDSC2 + DepMap data loading & merging
│   ├── feature_engineering.py   # Mutation encoding, HRD score, pathway features
│   ├── models.py                # Model training, CV, serialization
│   ├── evaluation.py            # Metrics, ROC/PR curves, reports
│   ├── biomarker_analysis.py    # SHAP, Mann-Whitney U, heatmaps
│   └── utils.py                 # Logging, I/O, statistical utilities
├── scripts/
│   ├── train.py                 # CLI: full training pipeline
│   ├── evaluate.py              # CLI: model evaluation
│   └── analyze_biomarkers.py    # CLI: SHAP + biomarker discovery
└── tests/
    ├── __init__.py
    └── test_pipeline.py         # Unit + integration tests
```

---

## Configuration

All pipeline parameters are managed in `config/config.py`. Key settings:

```python
from config.config import PipelineConfig

cfg = PipelineConfig()
print(cfg.drugs)       # ['olaparib', 'rucaparib', 'niraparib', 'AZD6738', 'VE-822']
print(cfg.ddr_genes)   # ['BRCA1', 'BRCA2', 'ATM', 'ATR', ...]
print(cfg.cv_folds)    # 5
```

---

## Citation

If you use this pipeline in your research, please cite:

```
DDR Biomarker & Drug Response Prediction Pipeline (2024).
Computational identification of DDR pathway biomarkers for
PARP/ATR inhibitor sensitivity prediction using GDSC2 and DepMap.
GitHub: https://github.com/your-org/ddr-biomarker-pipeline
```

**Key references:**
- Farmer et al. (2005). Targeting the DNA repair defect in BRCA mutant cells as a therapeutic strategy. *Nature*, 434, 917-921.
- Lord & Ashworth (2017). PARP inhibitors: Synthetic lethality in the clinic. *Science*, 355, 1152-1158.
- Shen et al. (2015). ARID1A deficiency impairs the DNA damage checkpoint and sensitizes cells to ATR inhibitors. *Cancer Cell*, 30, 477-491.
- Iorio et al. (2016). A Landscape of Pharmacogenomic Interactions in Cancer. *Cell*, 166, 740-754.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.

---

## License

MIT License. See LICENSE file for details.
