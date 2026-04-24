"""
DDR Biomarker Pipeline — discover DNA-damage-repair biomarkers of drug
sensitivity from cell-line pharmacogenomic data.
"""

from .biomarker_analysis import (
    compute_effect_size,
    run_shap_analysis,
    run_statistical_tests,
    summarize_biomarkers,
)
from .data_loader import generate_synthetic_data, merge_datasets
from .evaluation import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall,
    plot_roc_curves,
)
from .feature_engineering import (
    build_feature_matrix,
    compute_ddr_burden,
    compute_hrd_score,
    compute_msi_status,
)
from .models import (
    compare_models,
    evaluate_model,
    train_gradient_boosting,
    train_logistic_regression,
)
from .utils import Timer, ensure_dir, set_seed, setup_logging

__all__ = [
    "generate_synthetic_data",
    "merge_datasets",
    "build_feature_matrix",
    "compute_ddr_burden",
    "compute_hrd_score",
    "compute_msi_status",
    "train_gradient_boosting",
    "train_logistic_regression",
    "evaluate_model",
    "compare_models",
    "run_shap_analysis",
    "run_statistical_tests",
    "compute_effect_size",
    "summarize_biomarkers",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_precision_recall",
    "plot_roc_curves",
    "Timer",
    "ensure_dir",
    "set_seed",
    "setup_logging",
]
