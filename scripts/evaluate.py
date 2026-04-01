#!/usr/bin/env python3
"""
Evaluation CLI for the DDR Biomarker Pipeline.

Loads a saved model + test dataset, runs full evaluation:
  - AUC-ROC, AUC-PR, F1, accuracy
  - ROC and PR curve plots
  - Confusion matrix
  - Feature importance plot
  - Classification report

Usage
-----
    python scripts/evaluate.py \\
        --model-path results/olaparib/models/gradient_boosting_model.pkl \\
        --data-path results/olaparib/test_data.pkl \\
        --output-dir results/olaparib/evaluation \\
        --drug olaparib \\
        --plot-roc --plot-pr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import ModelEvaluator
from src.models import ModelTrainer
from src.utils import (
    ensure_dir,
    load_pickle,
    print_banner,
    save_dataframe,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DDR Biomarker model on test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model .pkl file (from train.py).",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to test_data.pkl (saved by train.py).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots and reports. Defaults to model directory.",
    )
    parser.add_argument(
        "--drug", type=str, default="",
        help="Drug name for plot titles.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binary predictions.",
    )
    parser.add_argument(
        "--plot-roc", action="store_true",
        help="Generate ROC curve plot.",
    )
    parser.add_argument(
        "--plot-pr", action="store_true",
        help="Generate Precision-Recall curve plot.",
    )
    parser.add_argument(
        "--plot-confusion", action="store_true",
        help="Generate confusion matrix plot.",
    )
    parser.add_argument(
        "--plot-importance", action="store_true",
        help="Generate feature importance plot.",
    )
    parser.add_argument(
        "--all-plots", action="store_true",
        help="Generate all available plots.",
    )
    parser.add_argument(
        "--top-n-features", type=int, default=20,
        help="Number of features to show in importance plot.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the evaluation pipeline."""
    args = parse_args(argv)

    setup_logging(log_level="DEBUG" if args.verbose else "INFO")
    print_banner(f"DDR BIOMARKER PIPELINE — Evaluation: {args.drug or 'Drug'}")

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    logger.info("Loading model from: %s", args.model_path)
    payload = ModelTrainer.load_model(args.model_path)

    model = payload["model"]
    model_name = payload.get("model_name", "model")
    feature_names = payload.get("feature_names", [])
    cv_results = payload.get("cv_results", {})

    logger.info("Model: %s", model_name)
    if cv_results:
        logger.info(
            "CV ROC-AUC: %.3f ± %.3f",
            cv_results.get("mean_roc_auc", 0),
            cv_results.get("std_roc_auc", 0),
        )

    # ---------------------------------------------------------------
    # 2. Load test data
    # ---------------------------------------------------------------
    logger.info("Loading test data from: %s", args.data_path)
    test_data = load_pickle(args.data_path)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    ic50_test = test_data.get("ic50_test", None)

    logger.info(
        "Test set: %d samples | %d sensitive / %d resistant",
        len(X_test), y_test.sum(), (y_test == 0).sum(),
    )

    # ---------------------------------------------------------------
    # 3. Set up output directory
    # ---------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model_path).parent.parent / "evaluation"
    ensure_dir(output_dir)

    # ---------------------------------------------------------------
    # 4. Generate predictions
    # ---------------------------------------------------------------
    import numpy as np

    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_arr)[:, 1]
    else:
        scores = model.decision_function(X_arr)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    y_pred = (y_prob >= args.threshold).astype(int)

    # ---------------------------------------------------------------
    # 5. Compute metrics
    # ---------------------------------------------------------------
    evaluator = ModelEvaluator(
        output_dir=str(output_dir),
        drug_name=args.drug,
    )

    metrics = evaluator.compute_all_metrics(
        y_test.values if hasattr(y_test, "values") else y_test,
        y_prob,
        y_pred,
        model_name=model_name,
        threshold=args.threshold,
    )

    logger.info("\nEvaluation Metrics:")
    logger.info("  ROC-AUC:    %.4f", metrics["roc_auc"])
    logger.info("  PR-AUC:     %.4f", metrics["pr_auc"])
    logger.info("  F1 (macro): %.4f", metrics["f1"])
    logger.info("  Accuracy:   %.4f", metrics["accuracy"])
    logger.info("  Kappa:      %.4f", metrics["kappa"])

    # ---------------------------------------------------------------
    # 6. Save metrics
    # ---------------------------------------------------------------
    save_results(metrics, output_dir, filename=f"test_metrics_{model_name}.json")

    # ---------------------------------------------------------------
    # 7. Generate plots
    # ---------------------------------------------------------------
    plot_flags = {
        "roc": args.plot_roc or args.all_plots,
        "pr": args.plot_pr or args.all_plots,
        "confusion": args.plot_confusion or args.all_plots,
        "importance": args.plot_importance or args.all_plots,
    }

    if not any(plot_flags.values()):
        # Default: generate ROC and confusion matrix
        plot_flags["roc"] = True
        plot_flags["confusion"] = True

    y_true_arr = y_test.values if hasattr(y_test, "values") else y_test

    if plot_flags["roc"]:
        logger.info("Generating ROC curve ...")
        evaluator.plot_roc_curves({model_name: y_prob}, y_true_arr)

    if plot_flags["pr"]:
        logger.info("Generating PR curve ...")
        evaluator.plot_pr_curves({model_name: y_prob}, y_true_arr)

    if plot_flags["confusion"]:
        logger.info("Generating confusion matrix ...")
        evaluator.plot_confusion_matrix(y_true_arr, y_pred, model_name=model_name)

    if plot_flags["importance"]:
        if feature_names:
            logger.info("Generating feature importance plot ...")
            evaluator.plot_feature_importance(
                model, feature_names, model_name=model_name,
                top_n=args.top_n_features,
            )
        else:
            logger.warning("No feature names in payload; skipping importance plot.")

    # ---------------------------------------------------------------
    # 8. Classification report
    # ---------------------------------------------------------------
    evaluator.generate_report(y_true_arr, y_pred, model_name=model_name)

    # ---------------------------------------------------------------
    # 9. Save predictions
    # ---------------------------------------------------------------
    import pandas as pd
    pred_df = pd.DataFrame({
        "cell_line": X_test.index if hasattr(X_test, "index") else range(len(y_true_arr)),
        "y_true": y_true_arr,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    if ic50_test is not None:
        pred_df["ln_ic50"] = ic50_test.values if hasattr(ic50_test, "values") else ic50_test
    save_dataframe(pred_df, output_dir, f"predictions_{model_name}.csv", index=False)

    print_banner("Evaluation Complete")
    logger.info("Results saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
