#!/usr/bin/env python3
"""Generate evaluation plots from pipeline results.

Usage:
    python scripts/run_evaluation.py --results-dir results --output-dir results/figures
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import plot_model_comparison, plot_roc_curves
from src.utils import ensure_dir, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results-dir", default="results", help="Directory with model results")
    parser.add_argument(
        "--output-dir", default="results/figures", help="Output directory for plots"
    )
    args = parser.parse_args()

    logger = setup_logging("evaluation")
    ensure_dir(args.output_dir)

    # Try loading pickled results first (saved by run_pipeline.py)
    results_path = os.path.join(args.results_dir, "model_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        logger.info("Loaded results from %s", results_path)
        plot_roc_curves(results, os.path.join(args.output_dir, "roc_curves.png"))
        plot_model_comparison(results, os.path.join(args.output_dir, "model_comparison.png"))
        logger.info("Plots saved to %s", args.output_dir)
    else:
        # Check if pipeline output CSVs exist
        csv_path = os.path.join(args.results_dir, "model_comparison.csv")
        if os.path.exists(csv_path):
            logger.info("Pickle not found; model_comparison.csv exists at %s", csv_path)
            logger.info(
                "Re-run the pipeline to generate full evaluation plots: python scripts/run_pipeline.py"
            )
        else:
            logger.error(
                "No results found in %s. Run the pipeline first: python scripts/run_pipeline.py",
                args.results_dir,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
