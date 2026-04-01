#!/usr/bin/env python3
"""
Biomarker analysis CLI for the DDR Biomarker Pipeline.

Loads a saved model + test data, runs SHAP-based feature attribution,
Mann-Whitney U biomarker association tests, Cohen's d effect sizes,
and generates a biomarker heatmap.

Usage
-----
    python scripts/analyze_biomarkers.py \\
        --model-path results/olaparib/models/gradient_boosting_model.pkl \\
        --data-path results/olaparib/test_data.pkl \\
        --drug olaparib \\
        --top-n 20 \\
        --output-dir results/olaparib/biomarkers

    # With synthetic data
    python scripts/analyze_biomarkers.py \\
        --synthetic --drug olaparib --output-dir results/test/biomarkers
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import PipelineConfig
from src.biomarker_analysis import BiomarkerAnalyzer
from src.data_loader import GDSCDataLoader
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import (
    ensure_dir,
    load_pickle,
    print_banner,
    save_dataframe,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP biomarker analysis for DDR drug sensitivity models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model + data
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to saved model .pkl. Required unless --synthetic.",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to test_data.pkl saved by train.py. Required unless --synthetic.",
    )

    # Analysis settings
    parser.add_argument(
        "--drug", type=str, default="olaparib",
        help="Drug name (for plots and reports).",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top biomarkers to report and plot.",
    )
    parser.add_argument(
        "--max-shap-samples", type=int, default=200,
        help="Maximum samples to use for SHAP computation.",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP analysis (faster, statistical tests only).",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip biomarker clustermap (can be slow for large N).",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots and CSVs.",
    )

    # Mode
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (for testing).",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=200,
        help="N cell lines for synthetic mode.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the biomarker analysis pipeline."""
    args = parse_args(argv)
    setup_logging(log_level="DEBUG" if args.verbose else "INFO")

    print_banner(f"DDR BIOMARKER ANALYSIS — {args.drug.upper()}")

    # ---------------------------------------------------------------
    # 1. Determine output directory
    # ---------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.model_path:
        output_dir = Path(args.model_path).parent.parent / "biomarkers"
    else:
        output_dir = Path("results") / args.drug.lower() / "biomarkers"
    ensure_dir(output_dir)
    logger.info("Output directory: %s", output_dir)

    # ---------------------------------------------------------------
    # 2. Load model + data
    # ---------------------------------------------------------------
    cfg = PipelineConfig()

    if args.synthetic:
        # Generate synthetic data and train a quick model
        logger.info("Generating synthetic data (n=%d) ...", args.n_synthetic)
        loader = GDSCDataLoader(cfg)
        gdsc2_df, mutations_df = loader.generate_synthetic_data(
            n_cell_lines=args.n_synthetic, drug=args.drug, seed=42
        )
        merged_df, ic50_series = loader.merge_datasets(gdsc2_df, mutations_df, drug=args.drug)

        fe = FeatureEngineer(cfg)
        X, y = fe.prepare_feature_matrix(merged_df, ic50_series, fit_scaler=True)

        # Train a gradient boosting model quickly
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        gb.fit(X.values, y.values)

        model = gb
        model_name = "gradient_boosting"
        feature_names = list(X.columns)
        X_test = X
        y_test = y
        ic50_continuous = ic50_series.reindex(X.index)

    else:
        # Load saved model
        if not args.model_path:
            logger.error("--model-path is required unless --synthetic is used.")
            return 1
        if not args.data_path:
            logger.error("--data-path is required unless --synthetic is used.")
            return 1

        logger.info("Loading model from: %s", args.model_path)
        payload = ModelTrainer.load_model(args.model_path)
        model = payload["model"]
        model_name = payload.get("model_name", "model")
        feature_names = payload.get("feature_names", [])

        logger.info("Loading test data from: %s", args.data_path)
        test_data = load_pickle(args.data_path)
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        ic50_continuous = test_data.get("ic50_test", None)

        if not feature_names:
            feature_names = list(X_test.columns) if hasattr(X_test, "columns") else []

    logger.info(
        "Test set: %d samples, %d features | %d sensitive / %d resistant",
        len(X_test), X_test.shape[1], y_test.sum(), (y_test == 0).sum(),
    )

    # ---------------------------------------------------------------
    # 3. Initialize analyzer
    # ---------------------------------------------------------------
    analyzer = BiomarkerAnalyzer(
        output_dir=str(output_dir),
        drug_name=args.drug,
        top_n=args.top_n,
    )

    # ---------------------------------------------------------------
    # 4. SHAP analysis
    # ---------------------------------------------------------------
    top_biomarkers_df = pd.DataFrame()
    shap_values = None

    if not args.no_shap:
        logger.info("Computing SHAP values ...")
        try:
            # Limit samples for speed
            n_shap = min(args.max_shap_samples, len(X_test))
            rng = np.random.default_rng(42)
            shap_idx = rng.choice(len(X_test), n_shap, replace=False)

            X_shap = X_test.iloc[shap_idx] if hasattr(X_test, "iloc") else X_test[shap_idx]

            shap_values = analyzer.compute_shap_values(
                model, X_shap,
                max_background_samples=min(100, n_shap),
            )

            # Identify top biomarkers
            top_biomarkers_df = analyzer.identify_top_biomarkers(
                shap_values, X_shap, top_n=args.top_n
            )
            logger.info(
                "Top biomarker: %s (mean |SHAP| = %.4f)",
                top_biomarkers_df.iloc[0]["feature"] if len(top_biomarkers_df) > 0 else "N/A",
                top_biomarkers_df.iloc[0]["mean_abs_shap"] if len(top_biomarkers_df) > 0 else 0.0,
            )

            # Save top biomarkers CSV
            if len(top_biomarkers_df) > 0:
                save_dataframe(
                    top_biomarkers_df, output_dir, "top_biomarkers_shap.csv", index=False
                )

            # SHAP summary plot
            logger.info("Generating SHAP summary plot ...")
            analyzer.plot_shap_summary(shap_values, X_shap, top_n=args.top_n)

        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)

    # ---------------------------------------------------------------
    # 5. Statistical association tests
    # ---------------------------------------------------------------
    association_df = pd.DataFrame()
    if ic50_continuous is not None:
        logger.info("Running Mann-Whitney U association tests ...")
        try:
            association_df = analyzer.test_biomarker_association(
                X_test, ic50_continuous
            )
            if len(association_df) > 0:
                n_sig = (association_df.get("q_value", association_df["p_value"]) < 0.05).sum()
                logger.info(
                    "Association tests: %d features, %d significant (q<0.05)",
                    len(association_df), n_sig,
                )
                save_dataframe(
                    association_df, output_dir, "biomarker_associations.csv", index=False
                )
        except Exception as e:
            logger.warning("Association testing failed: %s", e)

    # ---------------------------------------------------------------
    # 6. Effect sizes
    # ---------------------------------------------------------------
    effect_sizes_df = pd.DataFrame()
    if ic50_continuous is not None:
        logger.info("Computing Cohen's d effect sizes ...")
        try:
            effect_sizes_df = analyzer.compute_effect_sizes(X_test, ic50_continuous)
            if len(effect_sizes_df) > 0:
                save_dataframe(
                    effect_sizes_df, output_dir, "effect_sizes.csv", index=False
                )
        except Exception as e:
            logger.warning("Effect size computation failed: %s", e)

    # ---------------------------------------------------------------
    # 7. Biomarker heatmap
    # ---------------------------------------------------------------
    if not args.no_heatmap and ic50_continuous is not None:
        logger.info("Generating biomarker clustermap ...")
        try:
            top_feat = (
                top_biomarkers_df["feature"].tolist()[:args.top_n]
                if len(top_biomarkers_df) > 0 else None
            )
            analyzer.plot_biomarker_heatmap(
                X_test, ic50_continuous, top_features=top_feat
            )
        except Exception as e:
            logger.warning("Heatmap failed: %s", e)

    # ---------------------------------------------------------------
    # 8. Biomarker boxplots
    # ---------------------------------------------------------------
    if ic50_continuous is not None:
        logger.info("Generating biomarker boxplots ...")
        try:
            top_feat = (
                top_biomarkers_df["feature"].tolist()[:6]
                if len(top_biomarkers_df) > 0 else None
            )
            analyzer.plot_top_biomarker_boxplots(
                X_test, ic50_continuous, top_features=top_feat
            )
        except Exception as e:
            logger.warning("Boxplots failed: %s", e)

    # ---------------------------------------------------------------
    # 9. Generate summary report
    # ---------------------------------------------------------------
    logger.info("Generating biomarker report ...")
    analyzer.generate_biomarker_report(
        top_biomarkers_df, association_df, effect_sizes_df
    )

    print_banner("Biomarker Analysis Complete")
    logger.info("Results saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
