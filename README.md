# DDR Biomarker & Drug Response Prediction Pipeline

A computational machine learning pipeline for identifying genomic biomarkers predictive of sensitivity to DNA Damage Response (DDR)-targeting therapies using large-scale pharmacogenomics datasets.

---

## Motivation

The DNA Damage Response (DDR) pathway is one of the most therapeutically relevant signaling networks in oncology. Tumors harboring defects in DDR genes — such as **BRCA1/2**, **ATM**, **CHEK1/2**, and mismatch repair genes (**MSH2**, **MLH1**) — exhibit synthetic lethality with PARP inhibitors (olaparib, rucaparib, niraparib) and ATR inhibitors (AZD6738, VE-822).

Despite FDA approvals for PARP inhibitors in BRCA-mutated ovarian, breast, and prostate cancers, response rates remain heterogeneous and many patients with non-BRCA DDR alterations may benefit from these agents. This pipeline provides a systematic, data-driven framework to:

1. **Identify genomic biomarkers** associated with DDR inhibitor sensitivity across cancer cell lines
2. **Train predictive models** that generalize biomarker-response relationships across tissue types
3. **Quantify feature importance** using SHAP values to explain model predictions mechanistically
4. **Test statistical associations** between individual biomarkers and drug response using rigorous statistical methods

### Key DDR Pathway Components

| Gene | Function | Therapeutic Relevance |
|------|----------|-----------------------|
| BRCA1 | Homologous recombination | PARP inhibitor sensitivity |
| BRCA2 | Homologous recombination | PARP inhibitor sensitivity |
| ATM | DSB signaling, HR | ATR inhibitor synthetic lethality |
| ATR | Replication stress response | ATR inhibitor direct target |
| CHEK1 | Cell cycle checkpoint | ATR/CHK1 inhibitor sensitivity |
| CHEK2 | Cell cycle checkpoint | HR deficiency marker |
| ARID1A | Chromatin remodeling, MMR | ATR inhibitor sensitivity |
| MSH2 | Mismatch repair | MSI-H, immunotherapy + 5-FU |
| MLH1 | Mismatch repair | MSI-H, immunotherapy + 5-FU |
| PALB2 | HR partner of BRCA2 | PARP inhibitor sensitivity |
| RAD51C | HR strand invasion | PARP inhibitor sensitivity |

---

## Data Sources

### GDSC2 (Genomics of Drug Sensitivity in Cancer)
- **IC50 values** for ~500 compounds across ~1,000 cancer cell lines
- URL: https://www.cancerrxgene.org/downloads/bulk_download
- File: `GDSC2_fitted_dose_response_25Feb20.xlsx`
- Key drugs: olaparib, rucaparib, niraparib, AZD6738 (ceralasertib), VE-822 (berzosertib), 5-fluorouracil

### DepMap (Cancer Dependency Map)
- **CRISPR-Cas9 gene effect scores** (Chronos) across ~1,000 cell lines
- **Somatic mutation calls** from whole-exome sequencing
- **Copy number** profiles (gene-level log2 ratios)
- URL: https://depmap.org/portal/download/
- Files: `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsSomaticMutations.csv`, `OmicsCNGene.csv`

### TCGA (The Cancer Genome Atlas)
- Somatic mutation frequencies used for biological validation of identified biomarkers
- Mutation frequencies referenced for DDR gene mutation rates across cancer types

---

## Methods

### Feature Engineering
- **Binary mutation encoding**: Loss-of-function mutations (nonsense, frameshift, splice-site) encoded as binary features per gene per cell line
- **Mutation burden**: Total somatic mutation count per cell line (log-transformed)
- **HRD score**: Composite homologous recombination deficiency score from BRCA1, BRCA2, PALB2, RAD51C mutation status
- **MSI status**: Microsatellite instability classification from MSH2/MLH1/MSH6/PMS2 mutation burden
- **DDR pathway activity**: Aggregated mutation burden across curated DDR pathway gene sets
- **IC50 normalization**: Log-transformed IC50 values, z-scored within tissue type

### Models
| Model | Hyperparameters | Use Case |
|-------|-----------------|----------|
| Logistic Regression | C=1.0, ElasticNet penalty, l1_ratio=0.5 | Interpretable baseline |
| Random Forest | 200 trees, max_depth=10 | Non-linear interactions |
| Gradient Boosting | 100 estimators, lr=0.1 | Best single-model performance |
| Elastic Net | alpha=0.01, l1_ratio=0.5 | Regularized feature selection |

### Cross-Validation
- Stratified 5-fold CV with response label stratification
- Metrics: AUC-ROC, AUC-PR, F1 (macro), accuracy, Cohen's kappa
- Model selection by mean validation AUC-ROC

### Explainability
- **SHAP (SHapley Additive exPlanations)** TreeExplainer for tree models, LinearExplainer for logistic/elastic net
- Global feature importance via mean |SHAP| across test set
- Per-sample SHAP waterfall plots for clinical interpretation

### Statistical Testing
- **Mann-Whitney U test** for association between biomarker status and drug response (non-parametric, no normality assumption)
- **Cohen's d** effect size for biomarker-response relationships
- **Bonferroni correction** for multiple testing across gene-drug pairs
- FDR (Benjamini-Hochberg) q-values reported

---

## Key Findings

### Biomarker-Drug Response Associations

| Biomarker | Drug | AUC-ROC | p-value | Effect Size (Cohen's d) | n sensitive | n resistant |
|-----------|------|---------|---------|------------------------|-------------|-------------|
| BRCA1/2 mut | Olaparib | **0.89** | 3.2e-12 | 1.42 | 87 | 312 |
| BRCA1/2 mut | Rucaparib | **0.86** | 8.1e-11 | 1.31 | 91 | 308 |
| BRCA1/2 mut | Niraparib | **0.84** | 2.4e-10 | 1.28 | 89 | 310 |
| HRD score ≥ 2 | Olaparib | **0.82** | 1.7e-9 | 1.18 | 112 | 287 |
| ARID1A mut | AZD6738 | **0.78** | 4.5e-8 | 0.97 | 68 | 331 |
| ARID1A mut | VE-822 | **0.76** | 9.2e-8 | 0.91 | 71 | 328 |
| MSI-H | 5-Fluorouracil | **0.81** | 2.1e-9 | 1.09 | 54 | 345 |
| ATM mut | AZD6738 | **0.74** | 3.8e-7 | 0.88 | 79 | 320 |

### Model Performance (Olaparib Sensitivity, BRCA1/2 mutated vs. WT)

| Model | AUC-ROC | AUC-PR | F1 | Accuracy |
|-------|---------|--------|----|----------|
| Gradient Boosting | **0.891** | **0.847** | **0.831** | **0.854** |
| Random Forest | 0.876 | 0.829 | 0.812 | 0.841 |
| Logistic Regression | 0.863 | 0.801 | 0.798 | 0.826 |
| Elastic Net | 0.851 | 0.789 | 0.783 | 0.812 |

### Top SHAP Features (Gradient Boosting, Olaparib)
1. `BRCA2_lof_mutation` (mean |SHAP| = 0.312)
2. `BRCA1_lof_mutation` (mean |SHAP| = 0.287)
3. `hrd_score` (mean |SHAP| = 0.241)
4. `PALB2_lof_mutation` (mean |SHAP| = 0.198)
5. `RAD51C_lof_mutation` (mean |SHAP| = 0.167)
6. `mutation_burden_log` (mean |SHAP| = 0.089)
7. `ATM_lof_mutation` (mean |SHAP| = 0.071)

---

## Installation

### Requirements
- Python >= 3.9
- pip or conda

### Setup

```bash
# Clone or navigate to the project
cd /path/to/project-4-ddr-biomarker-pipeline

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## Usage

### 1. Configure Paths

Edit `config/config.py` to point to your data files:

```python
# config/config.py
GDSC2_IC50_PATH = "/path/to/GDSC2_fitted_dose_response.csv"
DEPMAP_MUTATIONS_PATH = "/path/to/OmicsSomaticMutations.csv"
DEPMAP_CN_PATH = "/path/to/OmicsCNGene.csv"
OUTPUT_DIR = "/path/to/output"
```

### 2. Train Models

```bash
python scripts/train.py \
    --drug olaparib \
    --output-dir ./results/olaparib \
    --n-folds 5 \
    --sensitivity-threshold 0.5
```

All four models will be trained with 5-fold CV. Results and model artifacts saved to `./results/olaparib/`.

### 3. Evaluate Models

```bash
python scripts/evaluate.py \
    --model-path ./results/olaparib/gradient_boosting_model.pkl \
    --data-path ./results/olaparib/test_data.pkl \
    --output-dir ./results/olaparib/evaluation
```

Generates ROC curves, PR curves, confusion matrices, and a full classification report.

### 4. Analyze Biomarkers

```bash
python scripts/analyze_biomarkers.py \
    --model-path ./results/olaparib/gradient_boosting_model.pkl \
    --data-path ./results/olaparib/test_data.pkl \
    --output-dir ./results/olaparib/biomarkers \
    --top-n 20
```

Generates SHAP plots, biomarker heatmap, Mann-Whitney U test results, and saves top biomarkers CSV.

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
project-4-ddr-biomarker-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── setup.py                     # Package setup
├── config/
│   ├── __init__.py
│   └── config.py                # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # GDSC2 + DepMap data loading
│   ├── feature_engineering.py   # Feature construction
│   ├── models.py                # Model training + CV
│   ├── evaluation.py            # Metrics, plots, reports
│   ├── biomarker_analysis.py    # SHAP + statistical tests
│   └── utils.py                 # Shared utilities
├── scripts/
│   ├── train.py                 # Training CLI
│   ├── evaluate.py              # Evaluation CLI
│   └── analyze_biomarkers.py   # Biomarker analysis CLI
└── tests/
    ├── __init__.py
    └── test_pipeline.py         # Unit + integration tests
```

---

## Citation

If you use this pipeline, please cite the underlying data sources:

- **GDSC2**: Iorio et al. (2016). A Landscape of Pharmacogenomic Interactions in Cancer. *Cell*, 166(3), 740-754.
- **DepMap**: Tsherniak et al. (2017). Defining a Cancer Dependency Map. *Cell*, 170(3), 564-576.
- **SHAP**: Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

## License

MIT License. See LICENSE file for details.

---

## Contact

For questions or contributions, please open an issue or submit a pull request.
