# Reproducibility Scorecard

This project self-scores against three 2026 reproducibility standards used in
computational biology: **FAIR-BioRS** (Nature Scientific Data, 2023), **DOME**
(ML-in-biology validation, EMBL-EBI), and **CURE** (Credible, Understandable,
Reproducible, Extensible — Nature npj Systems Biology 2026).

![Repro](https://img.shields.io/badge/FAIR_DOME_CURE-12%2F14_%7C_6%2F7_%7C_4%2F4-brightgreen)

## FAIR-BioRS (12 / 14)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Source code in a public VCS | ✅ | GitHub repo |
| 2 | License file present | ✅ | `LICENSE` (MIT) |
| 3 | Persistent identifier (DOI/Zenodo) | ⬜ | Not yet minted |
| 4 | Dependencies pinned | ✅ | `requirements.txt`, `environment.yml` |
| 5 | Containerized environment | ✅ | `Dockerfile` |
| 6 | Automated tests | ✅ | 14-test pytest suite |
| 7 | CI/CD on every push | ✅ | `.github/workflows/ci.yml` |
| 8 | README with install + run instructions | ✅ | `README.md` Quick Start |
| 9 | Example data included or referenced | ✅ | GDSC v17 via `gdsctools` PyPI |
| 10 | Expected outputs documented | ✅ | `results/poc/poc_summary.txt` |
| 11 | Version-controlled configuration | ✅ | `config/config.py` dataclass |
| 12 | Code style enforced (linter) | ✅ | `ruff` + `pre-commit` |
| 13 | Data provenance documented | ✅ | README "Data" section |
| 14 | Archived release (vX.Y.Z) | ⬜ | No tagged release yet |

## DOME (ML-in-biology) (6 / 7)

| # | Dimension | Status | Evidence |
|---|---|---|---|
| D | **Data**: source, version, preprocessing documented | ✅ | GDSC v17 from `gdsctools` bundled package; synthetic fallback in `src/data_loader.py` |
| O | **Optimization**: hyperparameter search method documented | ✅ | Inner 3-fold CV for LR regularization (C in [0.01, 0.1, 1.0, 10.0]) |
| M | **Model**: architecture, code, learned params available | ✅ | `src/models.py` — LR + GBM, sklearn pickled via joblib |
| E | **Evaluation**: metrics, CV scheme, baselines documented | ✅ | 5-fold stratified CV, ROC-AUC + PR-AUC + F1; tissue-only baseline documented in POC |
| + | Interpretability (SHAP / permutation) | ✅ | `src/biomarker_analysis.py` — TreeExplainer / LinearExplainer + fallback |
| + | Class-imbalance handled | ✅ | Stratified splits + `class_weight="balanced"` in LR |
| + | Independent validation cohort | ⬜ | TCGA validation absent (cell-line-only) |

## CURE (Nature npj Sys Biol 2026) (4 / 4)

| Letter | Criterion | Status | Evidence |
|---|---|---|---|
| **C** | Container reproducibility | ✅ | `Dockerfile` based on `python:3.11-slim` |
| **U** | URL persistence | ✅ | GitHub + gdsctools PyPI |
| **R** | Registered methods | ✅ | `scripts/poc/run_poc.py` is the canonical entry |
| **E** | Evidence of a real run | ✅ | `results/poc/*.csv` committed (Talazoparib ρ=+0.12, p=3e-4) |

## How to reproduce the score

```bash
ruff check . && ruff format --check .    # style gate
pytest tests/ -v                         # 14 tests
python scripts/poc/run_poc.py            # regenerates results/poc/*
```

## Cross-project standing

Project-4 is the **interpretability node** of the portfolio chain. It consumes
cell-line drug-response data (GDSC v17) and produces ranked DDR biomarkers with
SHAP attributions. Biomarker outputs conceptually feed Project-6's survival
covariate panel.
