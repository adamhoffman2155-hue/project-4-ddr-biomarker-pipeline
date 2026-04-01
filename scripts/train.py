#!/usr/bin/env python3
"""
Training CLI for the DDR Biomarker Pipeline.

Loads GDSC2 drug sensitivity + DepMap mutation data, engineers features,
trains four ML models (LogisticRegression, RandomForest, GradientBoosting,
ElasticNet) with stratified 5-fold CV, and saves model artifacts + results.

Usage
-----
    python scripts/train.py \\
        --gdsc2-path data/GDSC2_fitted_dose_response.csv \\
        --mutations-path data/OmicsSomaticMutations.csv \\
        --drug olaparib \\
        --output-dir results/olaparib \\
        --n-folds 5 \\
        --verbose

    # Use synthetic data for testing
    python scripts/train.py --drug olaparib --synthetic --output-dir results/test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import PipelineConfig
from src.data_loader import GDSCDataLoader
from src.evaluation import ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import (
    ensure_dir,
    get_timestamp,
    print_banner,
    save_dataframe,
    save_pickle,
    save_results,
    setup_logging,
    stratified_split,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDR Biomarker sensitivity prediction models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--gdsc2-path", type=str, default=None,
        help="Path to GDSC2 drug response CSV. Overrides config.",
    )
    data_group.add_argument(
        "--mutations-path", type=str, default=None,
        help="Path to DepMap somatic mutations CSV. Overrides config.",
    )
    data_group.add_argument(
        "--cn-path", type=str, default=None,
        help="(Optional) Path to DepMap copy number CSV.",
    )

    # Drug selection
    data_group.add_argument(
        "--drug", type=str, default="olaparib",
        help="Drug name to train on (case-insensitive GDSC2 match).",
    )
    data_group.add_argument(
        "--sensitivity-quantile", type=float, default=0.5,
        help="IC50 quantile threshold for sensitive/resistant binarization.",
    )

    # Training settings
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of stratified CV folds.",
    )
    train_group.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out as final test set.",
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output-dir", type=str, default="results",
        help="Root directory for saved artifacts.",
    )
    out_group.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (in addition to stdout).",
    )
    out_group.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating evaluation plots.",
    )

    # Mode
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (for testing / CI).",
    )
    mode_group.add_argument(
        "--n-synthetic", type=int, default=300,
        help="Number of synthetic cell lines if --synthetic is used.",
    )
    mode_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the training pipeline."""
    args = parse_args(argv)

    # -------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file=args.log_file)

    cfg = PipelineConfig()
    if args.n_folds:
        cfg.cv.n_folds = args.n_folds

    drug = args.drug
    output_dir = Path(args.output_dir) / drug.lower().replace(" ", "_")
    ensure_dir(output_dir)

    print_banner(f"DDR BIOMARKER PIPELINE — Training: {drug.upper()}")
    logger.info("Output directory: %s", output_dir)
    logger.info("Config: %d-fold CV, sensitivity quantile=%.2f",
                cfg.cv.n_folds, args.sensitivity_quantile)

    # -------------------------------------------------------------------
    # 2. Load data
    # -------------------------------------------------------------------
    loader = GDSCDataLoader(cfg)

    if args.synthetic:
        logger.info("Using SYNTHETIC data (n=%d cell lines)", args.n_synthetic)
        gdsc2_df, mutations_df = loader.generate_synthetic_data(
            n_cell_lines=args.n_synthetic,
            drug=cfg.resolve_drug_name(drug),
            seed=42,
        )
    else:
        if args.gdsc2_path:
            cfg.data.gdsc2_ic50_path = args.gdsc2_path
        if args.mutations_path:
            cfg.data.depmap_mutations_path = args.mutations_path

        logger.info("Loading GDSC2 data from %s", cfg.data.gdsc2_ic50_path)
        gdsc2_df = loader.load_gdsc2()

        logger.info("Loading DepMap mutations from %s", cfg.data.depmap_mutations_path)
        mutations_df = loader.load_depmap_mutations(genes=cfg.ddr_genes)

    # -------------------------------------------------------------------
    # 3. Merge datasets
    # -------------------------------------------------------------------
    cn_df = None
    if args.cn_path:
        cn_df = loader.load_depmap_copy_number(
            path=args.cn_path, genes=cfg.ddr_genes
        )

    logger.info("Merging datasets for drug: %s", drug)
    merged_df, ic50_series = loader.merge_datasets(
        gdsc2_df, mutations_df, drug=drug, cn_df=cn_df
    )

    if len(merged_df) < 30:
        logger.error(
            "Insufficient data after merge: %d samples. "
            "Minimum 30 required.", len(merged_df)
        )
        return 1

    # -------------------------------------------------------------------
    # 4. Feature engineering
    # -------------------------------------------------------------------
    logger.info("Engineering features ...")
    fe = FeatureEngineer(cfg)
    X, y = fe.prepare_feature_matrix(
        merged_df,
        ic50_series,
        sensitivity_quantile=args.sensitivity_quantile,
        fit_scaler=True,
        scale_features=True,
    )

    logger.info(
        "Feature matrix: %d samples x %d features | "
        "%d sensitive / %d resistant",
        X.shape[0], X.shape[1], y.sum(), (y == 0).sum(),
    )

    if X.shape[0] < 20:
        logger.error("Too few samples after feature engineering: %d", X.shape[0])
        return 1

    # -------------------------------------------------------------------
    # 5. Train / test split
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=args.test_size, random_state=cfg.cv.random_state
    )

    # Save test set for evaluation script
    test_data = {"X_test": X_test, "y_test": y_test, "ic50_test": ic50_series.reindex(X_test.index)}
    save_pickle(test_data, output_dir / "test_data.pkl")

    # -------------------------------------------------------------------
    # 6. Train all models with CV
    # -------------------------------------------------------------------
    print_banner("Model Training & Cross-Validation")
    trainer = ModelTrainer(cfg)
    cv_results = trainer.train_all_models(
        X_train, y_train,
        n_folds=cfg.cv.n_folds,
        verbose=True,
    )

    # -------------------------------------------------------------------
    # 7. Evaluate on held-out test set
    # -------------------------------------------------------------------
    print_banner("Test Set Evaluation")
    evaluator = ModelEvaluator(
        output_dir=str(output_dir / "evaluation"),
        drug_name=drug,
    )

    model_probs = {}
    model_preds = {}
    for model_name in trainer.MODEL_NAMES:
        y_prob = trainer.predict_proba(model_name, X_test)
        y_pred = trainer.predict(model_name, X_test)
        model_probs[model_name] = y_prob
        model_preds[model_name] = y_pred

    comparison_df = evaluator.compare_models(model_probs, y_test, model_preds)
    logger.info("\nTest set metrics:\n%s", comparison_df.to_string())

    if not args.no_plots:
        evaluator.plot_roc_curves(model_probs, y_test)
        evaluator.plot_pr_curves(model_probs, y_test)
        evaluator.plot_model_comparison(comparison_df)

        best_name, best_model = trainer.get_best_model()
        evaluator.plot_confusion_matrix(
            y_test, model_preds[best_name], model_name=best_name
        )
        evaluator.plot_feature_importance(
            best_model, X_train.columns.tolist(), model_name=best_name
        )

    # Generate report for best model
    best_name, _ = trainer.get_best_model()
    evaluator.generate_report(y_test, model_preds[best_name], model_name=best_name)

    # -------------------------------------------------------------------
    # 8. Save artifacts
    # -------------------------------------------------------------------
    print_banner("Saving Artifacts")

    # Save all models
    model_paths = trainer.save_all_models(str(output_dir / "models"))
    logger.info("Saved %d models to %s", len(model_paths), output_dir / "models")

    # Save CV summary
    cv_summary = trainer.get_cv_results_summary()
    save_dataframe(cv_summary, output_dir, "cv_results_summary.csv", index=True)

    # Save test metrics
    save_dataframe(comparison_df, output_dir, "test_metrics.csv", index=False)

    # Save feature matrix
    save_dataframe(X_train, output_dir, "train_features.csv")
    save_dataframe(X_test, output_dir, "test_features.csv")

    # Save run metadata
    run_meta = {
        "drug": drug,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_sensitive_train": int(y_train.sum()),
        "n_resistant_train": int((y_train == 0).sum()),
        "best_model": best_name,
        "best_roc_auc_cv": float(cv_results[best_name]["mean_roc_auc"]),
        "best_roc_auc_test": float(
            comparison_df.loc[comparison_df["model"] == best_name, "roc_auc"].iloc[0]
            if len(comparison_df) > 0 else 0.0
        ),
        "timestamp": get_timestamp(),
        "sensitivity_quantile": args.sensitivity_quantile,
        "n_folds": cfg.cv.n_folds,
    }
    save_results(run_meta, output_dir, "run_metadata.json")

    print_banner("Training Complete")
    logger.info("All artifacts saved to: %s", output_dir)
    logger.info(
        "Best model: %s | Test ROC-AUC: %.3f",
        best_name,
        run_meta["best_roc_auc_test"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
