#!/usr/bin/env python3
"""Generate evaluation plots for trained models.

Usage:
    python scripts/run_evaluation.py --results-dir results --output-dir results/figures
"""

import argparse
import os
import sys
import pickle
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import plot_roc_curves, plot_model_comparison
from src.utils import setup_logging, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results-dir", default="results", help="Directory with model results")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory for plots")
    args = parser.parse_args()

    logger = setup_logging("evaluation")
    ensure_dir(args.output_dir)

    # Load saved results if available
    results_path = os.path.join(args.results_dir, "model_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        logger.info(f"Loaded results from {results_path}")
        plot_roc_curves(results, os.path.join(args.output_dir, "roc_curves.png"))
        plot_model_comparison(results, os.path.join(args.output_dir, "model_comparison.png"))
        logger.info(f"Plots saved to {args.output_dir}")
    else:
        logger.error(f"No results found at {results_path}. Run the pipeline first: python scripts/run_pipeline.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
