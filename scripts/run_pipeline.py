#!/usr/bin/env python
"""
Full DDR Biomarker Discovery Pipeline.

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --n-cell-lines 300 --drugs olaparib rucaparib
"""

import argparse
import os
import pickle
import sys

# Ensure the project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config.config import PipelineConfig
from src.biomarker_analysis import (
    compute_effect_size,
    run_shap_analysis,
    run_statistical_tests,
    summarize_biomarkers,
)
from src.data_loader import generate_synthetic_data, merge_datasets
from src.evaluation import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall,
    plot_roc_curves,
)
from src.feature_engineering import build_feature_matrix
from src.models import (
    compare_models,
    train_gradient_boosting,
    train_logistic_regression,
)
from src.utils import Timer, set_seed, setup_logging

logger = setup_logging("pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DDR Biomarker Discovery Pipeline",
    )
    parser.add_argument(
        "--n-cell-lines",
        type=int,
        default=200,
        help="Number of synthetic cell lines (default: 200)",
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=None,
        help="Subset of drugs to analyse (default: all 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline."""
    args = parse_args()
    config = PipelineConfig()
    config.RANDOM_SEED = args.seed
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    set_seed(config.RANDOM_SEED)
    config.validate()
    logger.info("\n%s", config.summary())

    drugs = args.drugs or config.DDR_DRUGS

    # ------------------------------------------------------------------
    # 1. Generate / load data
    # ------------------------------------------------------------------
    with Timer("Data generation"):
        data = generate_synthetic_data(
            n_cell_lines=args.n_cell_lines,
            seed=config.RANDOM_SEED,
            ddr_genes=config.DDR_GENES,
            drug_names=config.DDR_DRUGS,
            tissue_types=config.TISSUE_TYPES,
            mutation_rate=config.MUTATION_RATE,
            sensitivity_boost=config.SENSITIVITY_BOOST,
        )
        ic50_df = data["ic50"]
        mutation_df = data["mutations"]
        metadata_df = data["metadata"]
        merged_df = merge_datasets(ic50_df, mutation_df, metadata_df)

    logger.info("Merged dataset shape: %s", merged_df.shape)

    # ------------------------------------------------------------------
    # 2. Per-drug modelling loop
    # ------------------------------------------------------------------
    all_results = {}

    for drug in drugs:
        logger.info("=" * 60)
        logger.info("Processing drug: %s", drug)
        logger.info("=" * 60)

        with Timer(f"Feature engineering ({drug})"):
            X, y = build_feature_matrix(mutation_df, ic50_df, drug, config)

        if (
            y.sum() < config.MIN_SAMPLES_PER_CLASS
            or (len(y) - y.sum()) < config.MIN_SAMPLES_PER_CLASS
        ):
            logger.warning("Skipping %s — insufficient class balance", drug)
            continue

        # Train models
        with Timer(f"Logistic Regression ({drug})"):
            lr_result = train_logistic_regression(X, y, config)

        with Timer(f"Gradient Boosting ({drug})"):
            gb_result = train_gradient_boosting(X, y, config)

        drug_results = {
            f"LR ({drug})": lr_result,
            f"GBM ({drug})": gb_result,
        }
        all_results.update(drug_results)

        # Compare
        compare_models(drug_results)

        # Biomarker analysis (use the better model)
        best_name = max(drug_results, key=lambda k: drug_results[k]["mean_metrics"]["auc"])
        best_result = drug_results[best_name]
        best_model = best_result["model"]

        with Timer(f"Biomarker analysis ({drug})"):
            feature_names = list(X.columns)
            shap_df = run_shap_analysis(best_model, X, feature_names)
            stats_df = run_statistical_tests(X, y, feature_names)
            effect_df = compute_effect_size(X, y, feature_names)
            biomarker_summary = summarize_biomarkers(shap_df, stats_df, effect_df)

        # Save biomarker table
        out_path = config.get_output_path(f"biomarkers_{drug}.csv")
        biomarker_summary.to_csv(out_path, index=False)
        logger.info("Biomarker summary saved to %s", out_path)

        # Plots
        plot_roc_curves(drug_results, config.get_output_path(f"roc_{drug}.png"))
        plot_precision_recall(drug_results, config.get_output_path(f"pr_{drug}.png"))
        plot_model_comparison(drug_results, config.get_output_path(f"comparison_{drug}.png"))

        # Confusion matrix for best model
        y_pred = best_model.predict(X)
        plot_confusion_matrix(
            y.values,
            y_pred,
            save_path=config.get_output_path(f"cm_{drug}.png"),
            title=f"Confusion Matrix — {best_name}",
        )

    # ------------------------------------------------------------------
    # 3. Global summary
    # ------------------------------------------------------------------
    if all_results:
        logger.info("=" * 60)
        logger.info("GLOBAL MODEL COMPARISON")
        logger.info("=" * 60)
        summary_df = compare_models(all_results)
        summary_df.to_csv(config.get_output_path("model_comparison.csv"))

        plot_roc_curves(all_results, config.get_output_path("roc_all.png"))
        plot_model_comparison(all_results, config.get_output_path("comparison_all.png"))

        # Save results for run_evaluation.py
        pkl_path = config.get_output_path("model_results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(all_results, f)
        logger.info("Model results saved to %s", pkl_path)

    logger.info("Pipeline complete. Results in %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
