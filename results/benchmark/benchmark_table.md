# Benchmark: DDR+MSI alone vs DDR+MSI + Tissue context

Five-fold stratified CV AUC per drug, two feature sets:
1. `DDR + MSI_FACTOR` — mutation + MSI only (12 features, no tissue)
2. `DDR + MSI + Tissue` — with tissue one-hot (39 features)

Source: `results/poc/per_drug_cv_auc.csv` (committed POC v2).

| Drug | n | n_sens | AUC (DDR+MSI) | AUC (+Tissue) | Δ AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Olaparib | 424 | 212 | 0.603 ± 0.051 | 0.736 ± 0.050 | +0.132 |
| AZD7762 | 424 | 212 | 0.594 ± 0.049 | 0.772 ± 0.036 | +0.178 |
| KU-55933 | 424 | 212 | 0.587 ± 0.054 | 0.750 ± 0.025 | +0.163 |
| Rucaparib | 460 | 230 | 0.522 ± 0.046 | 0.681 ± 0.046 | +0.159 |
| Talazoparib | 456 | 228 | 0.583 ± 0.052 | 0.713 ± 0.030 | +0.130 |

**Mean AUC (DDR+MSI only): 0.578**
**Mean AUC (+Tissue): 0.731**
**Mean Δ from tissue context: +0.153 AUC**

## Interpretation

DDR + MSI features alone discriminate sensitivity barely above chance
(mean AUC ≈ 0.58). Adding tissue-of-origin one-hots lifts mean AUC to
≈ 0.73, i.e. tissue context contributes ~0.15 AUC — the DDR/MSI signal
on cell lines is real but modest, and the headline AUC numbers in the
portfolio site reflect the full tissue-aware model. This is the
honest-negative framing called out in the README's Limits section.
