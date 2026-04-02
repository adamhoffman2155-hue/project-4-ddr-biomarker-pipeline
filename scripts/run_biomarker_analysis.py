#!/usr/bin/env python
"""
Standalone biomarker analysis script.

Generates synthetic data, builds features for a single drug, trains the
best model, and runs the full biomarker discovery workflow.

Usage::

    python scripts/run_biomarker_analysis.py
    python scripts/run_biomarker_analysis.py --drug olaparib
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config.config import PipelineConfig
from src.data_loader import generate_synthetic_data
from src.feature_engineering import build_feature_matrix
from src.models import train_gradient_boosting
from src.biomarker_analysis import (
    run_shap_analysis,
    run_statistical_tests,
    compute_effect_size,
    summarize_biomarkers,
)
from src.utils import setup_logging, set_seed

logger = setup_logging("biomarker_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="DDR Biomarker Analysis")
    parser.add_argument("--drug", type=str, default="olaparib",
                        help="Drug to analyse (default: olaparib)")
    args = parser.parse_args()

    config = PipelineConfig()
    set_seed(config.RANDOM_SEED)

    data = generate_synthetic_data(seed=config.RANDOM_SEED)
    X, y = build_feature_matrix(data["mutations"], data["ic50"], args.drug, config)

    result = train_gradient_boosting(X, y, config)
    model = result["model"]

    feature_names = list(X.columns)
    shap_df = run_shap_analysis(model, X, feature_names)
    stats_df = run_statistical_tests(X, y, feature_names)
    effect_df = compute_effect_size(X, y, feature_names)
    summary = summarize_biomarkers(shap_df, stats_df, effect_df)

    out_path = config.get_output_path(f"biomarkers_{args.drug}_standalone.csv")
    summary.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
