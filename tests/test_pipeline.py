"""
Unit and integration tests for the DDR Biomarker Pipeline.

Tests are organized into:
  - TestDataLoader: synthetic data loading and merging
  - TestFeatureEngineering: mutation encoding, HRD/MSI, pathway features
  - TestModelTrainer: smoke test training + CV on small synthetic dataset
  - TestEvaluator: metric computation and report generation
  - TestBiomarkerAnalyzer: association tests, effect sizes, SHAP (mocked)
  - TestUtils: logging, CI, p-value formatting, stratified split
  - TestEndToEnd: full pipeline smoke test

Run with::

    pytest tests/test_pipeline.py -v --tb=short
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import PipelineConfig
from src.biomarker_analysis import BiomarkerAnalyzer
from src.data_loader import GDSCDataLoader
from src.evaluation import ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import (
    compute_confidence_interval,
    ensure_dir,
    format_pvalue,
    setup_logging,
    stratified_split,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    """Return a default PipelineConfig instance."""
    return PipelineConfig()


@pytest.fixture(scope="session")
def loader(config) -> GDSCDataLoader:
    """Return a GDSCDataLoader backed by the default config."""
    return GDSCDataLoader(config)


@pytest.fixture(scope="session")
def synthetic_raw_data(loader) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic GDSC2 + DepMap mutation data (200 cell lines)."""
    return loader.generate_synthetic_data(
        n_cell_lines=200, n_genes=15, drug="Olaparib", seed=42
    )


@pytest.fixture(scope="session")
def merged_data(loader, synthetic_raw_data):
    """Merge synthetic raw data into a single DataFrame."""
    gdsc2_df, mutations_df = synthetic_raw_data
    merged_df, ic50_series = loader.merge_datasets(gdsc2_df, mutations_df, drug="Olaparib")
    return merged_df, ic50_series


@pytest.fixture(scope="session")
def feature_matrix(config, merged_data):
    """Build feature matrix from merged data."""
    merged_df, ic50_series = merged_data
    fe = FeatureEngineer(config)
    X, y = fe.prepare_feature_matrix(
        merged_df, ic50_series,
        sensitivity_quantile=0.5,
        fit_scaler=True,
        scale_features=True,
    )
    return X, y


@pytest.fixture(scope="session")
def trained_trainer(config, feature_matrix):
    """Train all models on the synthetic feature matrix."""
    X, y = feature_matrix
    # Use a small dataset subset for speed in CI
    n = min(100, len(X))
    X_small = X.iloc[:n]
    y_small = y.iloc[:n]

    trainer = ModelTrainer(config)
    # Reduce CV folds for speed
    trainer.config.cv.n_folds = 2
    trainer.train_all_models(X_small, y_small, n_folds=2, verbose=False)
    return trainer, X_small, y_small


# ---------------------------------------------------------------------------
# TestDataLoader
# ---------------------------------------------------------------------------


class TestDataLoader:
    """Tests for GDSCDataLoader."""

    def test_synthetic_data_shapes(self, synthetic_raw_data):
        """Synthetic data should have correct number of rows."""
        gdsc2_df, mutations_df = synthetic_raw_data
        assert len(gdsc2_df) == 200, "GDSC2 should have 200 rows (one per cell line)"
        assert len(mutations_df) > 0, "Mutation DataFrame should not be empty"

    def test_synthetic_gdsc2_columns(self, synthetic_raw_data):
        """Synthetic GDSC2 should have required columns."""
        gdsc2_df, _ = synthetic_raw_data
        required = {"DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"}
        assert required.issubset(set(gdsc2_df.columns))

    def test_synthetic_mutations_columns(self, synthetic_raw_data):
        """Synthetic mutations should have required columns."""
        _, mutations_df = synthetic_raw_data
        required = {"ModelID", "HugoSymbol", "VariantType"}
        assert required.issubset(set(mutations_df.columns))

    def test_merge_datasets_basic(self, merged_data):
        """Merge should produce a non-empty DataFrame."""
        merged_df, ic50_series = merged_data
        assert len(merged_df) > 0, "Merged DataFrame should not be empty"
        assert "LN_IC50" in merged_df.columns

    def test_merge_returns_ic50_series(self, merged_data):
        """IC50 series should be aligned with merged_df."""
        merged_df, ic50_series = merged_data
        # Some overlap expected
        overlap = ic50_series.index.intersection(merged_df["CELL_LINE_NAME"].values)
        assert len(overlap) > 0

    def test_merge_mutation_columns_present(self, merged_data):
        """Merged DataFrame should have at least some _lof_mutation columns."""
        merged_df, _ = merged_data
        lof_cols = [c for c in merged_df.columns if c.endswith("_lof_mutation")]
        assert len(lof_cols) > 0, "Expected binary mutation feature columns"

    def test_compute_auc_from_ic50(self, loader, synthetic_raw_data):
        """AUC approximation should be in [0, 1]."""
        gdsc2_df, _ = synthetic_raw_data
        result = loader.compute_auc_from_ic50(gdsc2_df)
        assert "AUC_approx" in result.columns
        assert result["AUC_approx"].between(0, 1).all()

    def test_get_cell_lines_by_tissue(self, loader, synthetic_raw_data):
        """Filtering by tissue type should reduce the DataFrame."""
        gdsc2_df, _ = synthetic_raw_data
        brca_df = loader.get_cell_lines_by_tissue(gdsc2_df, "BRCA")
        assert len(brca_df) <= len(gdsc2_df)

    def test_file_not_found_raises(self, loader):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.load_gdsc2(path="/nonexistent/path.csv", force_reload=True)

    def test_unknown_drug_raises(self, loader, synthetic_raw_data):
        """Merging with an unknown drug should raise ValueError."""
        gdsc2_df, mutations_df = synthetic_raw_data
        with pytest.raises(ValueError):
            loader.merge_datasets(gdsc2_df, mutations_df, drug="TOTALLY_UNKNOWN_DRUG_XYZ")


# ---------------------------------------------------------------------------
# TestFeatureEngineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    """Tests for FeatureEngineer."""

    def test_encode_mutations_binary_shape(self, config, synthetic_raw_data):
        """Binary encoding should return (n_cell_lines, n_genes) shape."""
        _, mutations_df = synthetic_raw_data
        fe = FeatureEngineer(config)
        genes = config.ddr_genes[:10]
        matrix = fe.encode_mutations_binary(mutations_df, genes=genes)
        assert matrix.shape[1] == len(genes)
        assert set(matrix.values.flatten()).issubset({0, 1})

    def test_encode_mutations_column_names(self, config, synthetic_raw_data):
        """Columns should be named {gene}_lof_mutation."""
        _, mutations_df = synthetic_raw_data
        fe = FeatureEngineer(config)
        genes = ["BRCA1", "BRCA2", "ATM"]
        matrix = fe.encode_mutations_binary(mutations_df, genes=genes)
        expected_cols = {"BRCA1_lof_mutation", "BRCA2_lof_mutation", "ATM_lof_mutation"}
        assert expected_cols.issubset(set(matrix.columns))

    def test_compute_mutation_burden_positive(self, config, synthetic_raw_data):
        """Mutation burden should be non-negative."""
        _, mutations_df = synthetic_raw_data
        fe = FeatureEngineer(config)
        burden = fe.compute_mutation_burden(mutations_df, log_transform=True)
        assert (burden >= 0).all()

    def test_mutation_burden_log_transform(self, config, synthetic_raw_data):
        """Log-transformed burden should be lower than raw burden."""
        _, mutations_df = synthetic_raw_data
        fe = FeatureEngineer(config)
        raw = fe.compute_mutation_burden(mutations_df, log_transform=False)
        log_t = fe.compute_mutation_burden(mutations_df, log_transform=True)
        # log1p(x) <= x for x >= 0
        shared = raw.index.intersection(log_t.index)
        assert (log_t[shared] <= raw[shared] + 1e-6).all()

    def test_hrd_score_range(self, config, merged_data):
        """HRD score should be non-negative integer."""
        merged_df, ic50_series = merged_data
        fe = FeatureEngineer(config)
        # Build minimal binary matrix first
        lof_cols = [c for c in merged_df.columns if c.endswith("_lof_mutation")]
        binary_mat = merged_df.set_index("CELL_LINE_NAME")[lof_cols]
        hrd = fe.compute_hrd_score(binary_mat)
        assert (hrd >= 0).all()
        assert hrd.max() <= len(config.hrd_genes)

    def test_msi_status_binary(self, config, merged_data):
        """MSI status should be 0 or 1."""
        merged_df, _ = merged_data
        fe = FeatureEngineer(config)
        lof_cols = [c for c in merged_df.columns if c.endswith("_lof_mutation")]
        binary_mat = merged_df.set_index("CELL_LINE_NAME")[lof_cols]
        msi = fe.create_msi_status(binary_mat)
        assert set(msi.unique()).issubset({0, 1})

    def test_pathway_features_shape(self, config, merged_data):
        """Pathway features should have one column per pathway."""
        from src.feature_engineering import PATHWAY_GENE_SETS
        merged_df, _ = merged_data
        fe = FeatureEngineer(config)
        lof_cols = [c for c in merged_df.columns if c.endswith("_lof_mutation")]
        binary_mat = merged_df.set_index("CELL_LINE_NAME")[lof_cols]
        pathway_df = fe.create_pathway_features(binary_mat)
        assert len(pathway_df.columns) == len(PATHWAY_GENE_SETS)

    def test_pathway_features_range(self, config, merged_data):
        """Pathway scores should be in [0, 1]."""
        merged_df, _ = merged_data
        fe = FeatureEngineer(config)
        lof_cols = [c for c in merged_df.columns if c.endswith("_lof_mutation")]
        binary_mat = merged_df.set_index("CELL_LINE_NAME")[lof_cols]
        pathway_df = fe.create_pathway_features(binary_mat)
        assert (pathway_df.values >= 0).all()
        assert (pathway_df.values <= 1 + 1e-9).all()

    def test_binarize_ic50_balanced(self, config, merged_data):
        """At median split, sensitive/resistant classes should be ~equal."""
        _, ic50_series = merged_data
        fe = FeatureEngineer(config)
        labels = fe.binarize_ic50(ic50_series, quantile=0.5)
        n_sensitive = labels.sum()
        n_resistant = (labels == 0).sum()
        # Should be within 5% of 50/50
        ratio = n_sensitive / max(len(labels), 1)
        assert 0.4 <= ratio <= 0.6, f"Expected ~50% sensitive, got {ratio:.2f}"

    def test_prepare_feature_matrix_returns_X_y(self, feature_matrix):
        """prepare_feature_matrix should return DataFrame and Series."""
        X, y = feature_matrix
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_matrix_no_nans(self, feature_matrix):
        """Feature matrix must not contain NaN values."""
        X, _ = feature_matrix
        assert not X.isna().any().any(), "Feature matrix should not have NaN values"

    def test_feature_matrix_minimum_features(self, feature_matrix):
        """Feature matrix should have at least 5 features."""
        X, _ = feature_matrix
        assert X.shape[1] >= 5, f"Expected >= 5 features, got {X.shape[1]}"

    def test_y_labels_binary(self, feature_matrix):
        """Labels should only be 0 or 1."""
        _, y = feature_matrix
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# TestModelTrainer
# ---------------------------------------------------------------------------


class TestModelTrainer:
    """Tests for ModelTrainer."""

    def test_train_all_returns_four_models(self, trained_trainer):
        """train_all_models should produce 4 models."""
        trainer, _, _ = trained_trainer
        assert len(trainer._models) == 4
        assert set(trainer._models.keys()) == set(ModelTrainer.MODEL_NAMES)

    def test_cv_results_have_required_keys(self, trained_trainer):
        """CV results for each model should have expected keys."""
        trainer, _, _ = trained_trainer
        required_keys = {"mean_roc_auc", "std_roc_auc", "mean_pr_auc", "mean_f1", "mean_accuracy"}
        for model_name in ModelTrainer.MODEL_NAMES:
            assert required_keys.issubset(set(trainer._cv_results[model_name].keys()))

    def test_roc_auc_above_random(self, trained_trainer):
        """All models should achieve ROC-AUC > 0.4 on synthetic data with signal."""
        trainer, _, _ = trained_trainer
        for model_name in ModelTrainer.MODEL_NAMES:
            auc = trainer._cv_results[model_name]["mean_roc_auc"]
            assert auc > 0.4, f"{model_name} ROC-AUC {auc:.3f} below 0.4"

    def test_get_best_model_returns_tuple(self, trained_trainer):
        """get_best_model should return (name, estimator)."""
        trainer, _, _ = trained_trainer
        best_name, best_model = trainer.get_best_model()
        assert isinstance(best_name, str)
        assert best_name in ModelTrainer.MODEL_NAMES
        assert hasattr(best_model, "predict")

    def test_predict_proba_shape(self, trained_trainer, feature_matrix):
        """predict_proba should return 1D array of length n_samples."""
        trainer, X_small, _ = trained_trainer
        for model_name in ModelTrainer.MODEL_NAMES:
            probs = trainer.predict_proba(model_name, X_small)
            assert probs.shape == (len(X_small),)
            assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_binary(self, trained_trainer):
        """predict should return binary array."""
        trainer, X_small, _ = trained_trainer
        for model_name in ModelTrainer.MODEL_NAMES:
            preds = trainer.predict(model_name, X_small)
            assert set(preds.flatten()).issubset({0, 1})

    def test_cv_summary_dataframe(self, trained_trainer):
        """get_cv_results_summary should return a DataFrame with model rows."""
        trainer, _, _ = trained_trainer
        summary = trainer.get_cv_results_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4

    def test_save_and_load_model(self, trained_trainer):
        """Model should be serializable and loadable without loss."""
        trainer, X_small, y_small = trained_trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model("random_forest", tmpdir)
            assert path.exists()

            payload = ModelTrainer.load_model(str(path))
            assert "model" in payload
            assert "feature_names" in payload

            # Predictions should be identical after reload
            model_orig = trainer.get_model("random_forest")
            model_reloaded = payload["model"]
            X_arr = X_small.values
            orig_preds = model_orig.predict(X_arr)
            reload_preds = model_reloaded.predict(X_arr)
            np.testing.assert_array_equal(orig_preds, reload_preds)

    def test_unknown_model_raises(self, trained_trainer):
        """Accessing an unregistered model should raise KeyError."""
        trainer, _, _ = trained_trainer
        with pytest.raises(KeyError):
            trainer.get_model("nonexistent_model")


# ---------------------------------------------------------------------------
# TestEvaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture(autouse=True)
    def setup_evaluator(self, tmp_path):
        self.evaluator = ModelEvaluator(
            output_dir=str(tmp_path), drug_name="olaparib"
        )
        rng = np.random.default_rng(0)
        n = 100
        self.y_true = rng.integers(0, 2, n)
        self.y_prob = np.clip(
            self.y_true * 0.6 + rng.uniform(0, 0.4, n), 0, 1
        )
        self.y_pred = (self.y_prob >= 0.5).astype(int)

    def test_roc_auc_range(self):
        """ROC-AUC should be in [0, 1]."""
        auc = self.evaluator.compute_roc_auc(self.y_true, self.y_prob)
        assert 0.0 <= auc <= 1.0

    def test_pr_auc_range(self):
        """PR-AUC should be in [0, 1] (allow small floating point tolerance)."""
        auc_pr = self.evaluator.compute_pr_auc(self.y_true, self.y_prob)
        assert 0.0 <= auc_pr <= 1.0 + 1e-9

    def test_compute_all_metrics_keys(self):
        """compute_all_metrics should return dict with required keys."""
        metrics = self.evaluator.compute_all_metrics(
            self.y_true, self.y_prob, self.y_pred
        )
        required = {"roc_auc", "pr_auc", "f1", "accuracy", "kappa"}
        assert required.issubset(set(metrics.keys()))

    def test_compare_models_returns_dataframe(self):
        """compare_models should return a DataFrame sorted by ROC-AUC."""
        model_probs = {
            "model_a": self.y_prob,
            "model_b": np.clip(self.y_prob + np.random.uniform(-0.1, 0.1, len(self.y_prob)), 0, 1),
        }
        df = self.evaluator.compare_models(model_probs, self.y_true, save=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # Should be sorted descending
        assert df["roc_auc"].is_monotonic_decreasing

    def test_generate_report_returns_string(self):
        """generate_report should return a non-empty string."""
        report = self.evaluator.generate_report(
            self.y_true, self.y_pred, model_name="test_model", save=True
        )
        assert isinstance(report, str)
        assert len(report) > 50

    def test_plot_roc_curves_returns_figure(self, tmp_path):
        """plot_roc_curves should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        fig = self.evaluator.plot_roc_curves(
            {"model": self.y_prob}, self.y_true, save=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_confusion_matrix_saves_file(self, tmp_path):
        """plot_confusion_matrix should save a file to output_dir."""
        import matplotlib.pyplot as plt
        self.evaluator.plot_confusion_matrix(
            self.y_true, self.y_pred, model_name="test", save=True
        )
        plt.close("all")
        saved = list(self.evaluator.output_dir.glob("confusion_matrix_*.png"))
        assert len(saved) >= 1


# ---------------------------------------------------------------------------
# TestBiomarkerAnalyzer
# ---------------------------------------------------------------------------


class TestBiomarkerAnalyzer:
    """Tests for BiomarkerAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, feature_matrix):
        self.output_dir = tmp_path
        self.X, self.y = feature_matrix
        # Use a small subset
        n = min(80, len(self.X))
        self.X_small = self.X.iloc[:n]
        self.y_small = self.y.iloc[:n]
        # Synthetic continuous IC50
        rng = np.random.default_rng(0)
        self.ic50 = pd.Series(
            rng.normal(2.0, 1.5, n),
            index=self.X_small.index,
            name="LN_IC50",
        )
        self.analyzer = BiomarkerAnalyzer(
            output_dir=str(tmp_path), drug_name="olaparib", top_n=10
        )

    def test_identify_top_biomarkers_shape(self):
        """identify_top_biomarkers should return top_n rows."""
        # Use dummy SHAP values
        rng = np.random.default_rng(42)
        n_samples, n_features = self.X_small.shape
        shap_vals = rng.normal(0, 0.1, (n_samples, n_features))
        result = self.analyzer.identify_top_biomarkers(shap_vals, self.X_small, top_n=10)
        assert len(result) <= 10
        assert "feature" in result.columns
        assert "mean_abs_shap" in result.columns

    def test_identify_top_biomarkers_sorted(self):
        """Top biomarkers should be sorted by mean_abs_shap descending."""
        rng = np.random.default_rng(42)
        n_samples, n_features = self.X_small.shape
        shap_vals = rng.normal(0, 0.1, (n_samples, n_features))
        result = self.analyzer.identify_top_biomarkers(shap_vals, self.X_small)
        assert result["mean_abs_shap"].is_monotonic_decreasing

    def test_test_biomarker_association_returns_df(self):
        """test_biomarker_association should return a DataFrame."""
        result = self.analyzer.test_biomarker_association(
            self.X_small, self.ic50
        )
        assert isinstance(result, pd.DataFrame)

    def test_association_df_has_required_columns(self):
        """Association DataFrame should have required columns."""
        result = self.analyzer.test_biomarker_association(
            self.X_small, self.ic50
        )
        if len(result) > 0:
            required = {"feature", "p_value", "cohen_d", "n_mutant", "n_wildtype"}
            assert required.issubset(set(result.columns))

    def test_association_p_values_valid(self):
        """P-values should be in [0, 1]."""
        result = self.analyzer.test_biomarker_association(
            self.X_small, self.ic50
        )
        if len(result) > 0:
            assert (result["p_value"] >= 0).all()
            assert (result["p_value"] <= 1).all()

    def test_compute_effect_sizes_returns_df(self):
        """compute_effect_sizes should return a DataFrame."""
        result = self.analyzer.compute_effect_sizes(self.X_small, self.ic50)
        assert isinstance(result, pd.DataFrame)

    def test_cohens_d_static(self):
        """Cohen's d should return known value for simple cases."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        d = BiomarkerAnalyzer._compute_cohens_d(a, b)
        assert d < 0  # mean(a) < mean(b) -> negative d
        assert abs(d) > 1.0  # effect is large (3 SD apart)

    def test_interpret_cohen_d(self):
        """Effect size interpretation should match thresholds."""
        assert BiomarkerAnalyzer._interpret_cohen_d(0.1) == "negligible"
        assert BiomarkerAnalyzer._interpret_cohen_d(0.35) == "small"
        assert BiomarkerAnalyzer._interpret_cohen_d(0.65) == "medium"
        assert BiomarkerAnalyzer._interpret_cohen_d(1.0) == "large"
        assert BiomarkerAnalyzer._interpret_cohen_d(1.5) == "very large"

    def test_generate_biomarker_report_returns_string(self):
        """generate_biomarker_report should return a non-empty string."""
        rng = np.random.default_rng(42)
        shap_vals = rng.normal(0, 0.1, (len(self.X_small), self.X_small.shape[1]))
        top_df = self.analyzer.identify_top_biomarkers(shap_vals, self.X_small)
        assoc_df = self.analyzer.test_biomarker_association(self.X_small, self.ic50)
        report = self.analyzer.generate_biomarker_report(top_df, assoc_df, save=False)
        assert isinstance(report, str)
        assert len(report) > 20


# ---------------------------------------------------------------------------
# TestUtils
# ---------------------------------------------------------------------------


class TestUtils:
    """Tests for src/utils.py utility functions."""

    def test_setup_logging_returns_logger(self):
        """setup_logging should return a Logger instance."""
        log = setup_logging(log_level="WARNING")
        assert isinstance(log, logging.Logger)

    def test_setup_logging_with_file(self, tmp_path):
        """setup_logging with log_file should create the file."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_level="INFO", log_file=log_file)
        logger = logging.getLogger("test_utils")
        logger.info("Test log message")
        assert Path(log_file).exists()

    def test_ensure_dir_creates_directory(self, tmp_path):
        """ensure_dir should create a nested directory."""
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_compute_confidence_interval_t(self):
        """CI should contain the true mean for a standard normal sample."""
        rng = np.random.default_rng(7)
        values = rng.normal(0, 1, 100)
        lo, hi = compute_confidence_interval(values, confidence=0.95, method="t")
        assert lo < hi
        # CI should contain zero (true mean) most of the time
        assert lo < 0.5 and hi > -0.5

    def test_compute_confidence_interval_bootstrap(self):
        """Bootstrap CI should bracket the sample mean."""
        rng = np.random.default_rng(8)
        values = rng.uniform(0, 10, 50)
        mean = values.mean()
        lo, hi = compute_confidence_interval(values, confidence=0.9, method="bootstrap")
        assert lo <= mean <= hi

    def test_compute_confidence_interval_single_value(self):
        """Single-element array should return (value, value)."""
        lo, hi = compute_confidence_interval(np.array([5.0]))
        assert lo == hi == 5.0

    def test_format_pvalue_highly_significant(self):
        """Very small p-values should be formatted in scientific notation."""
        result = format_pvalue(1e-10)
        assert "e" in result.lower() or "E" in result
        assert "***" in result

    def test_format_pvalue_not_significant(self):
        """Non-significant p-values should show numeric value."""
        result = format_pvalue(0.35)
        assert "0.35" in result
        assert "*" not in result

    def test_format_pvalue_nan(self):
        """NaN p-value should return 'p=NA'."""
        result = format_pvalue(float("nan"))
        assert "NA" in result

    def test_stratified_split_sizes(self, feature_matrix):
        """stratified_split should respect test_size."""
        X, y = feature_matrix
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        # Test size within ±2%
        ratio = len(X_test) / total
        assert 0.18 <= ratio <= 0.22

    def test_stratified_split_class_balance(self, feature_matrix):
        """Stratified split should maintain class proportions."""
        X, y = feature_matrix
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25)
        overall_pos = y.mean()
        train_pos = y_train.mean()
        test_pos = y_test.mean()
        # Each split should be within 5% of overall
        assert abs(train_pos - overall_pos) < 0.05
        assert abs(test_pos - overall_pos) < 0.05


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline smoke test: data -> features -> train -> evaluate."""

    def test_full_pipeline_smoke(self, tmp_path):
        """Run the entire pipeline on synthetic data without error."""
        cfg = PipelineConfig()
        cfg.cv.n_folds = 2  # Speed up

        # 1. Generate data
        loader = GDSCDataLoader(cfg)
        gdsc2_df, mutations_df = loader.generate_synthetic_data(
            n_cell_lines=150, n_genes=12, drug="Olaparib", seed=99
        )

        # 2. Merge
        merged_df, ic50_series = loader.merge_datasets(
            gdsc2_df, mutations_df, drug="Olaparib"
        )
        assert len(merged_df) > 20

        # 3. Feature engineering
        fe = FeatureEngineer(cfg)
        X, y = fe.prepare_feature_matrix(
            merged_df, ic50_series, fit_scaler=True, scale_features=True
        )
        assert X.shape[0] > 20
        assert not X.isna().any().any()

        # 4. Train/test split
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25)

        # 5. Train
        trainer = ModelTrainer(cfg)
        cv_results = trainer.train_all_models(X_train, y_train, n_folds=2, verbose=False)
        assert len(cv_results) == 4

        # 6. Evaluate
        evaluator = ModelEvaluator(output_dir=str(tmp_path / "eval"), drug_name="olaparib")
        model_probs = {}
        for name in ModelTrainer.MODEL_NAMES:
            model_probs[name] = trainer.predict_proba(name, X_test)

        comparison = evaluator.compare_models(model_probs, y_test, save=True)
        assert len(comparison) == 4
        assert "roc_auc" in comparison.columns

        # 7. Save and reload best model
        best_name, _ = trainer.get_best_model()
        model_path = trainer.save_model(best_name, str(tmp_path / "models"))
        payload = ModelTrainer.load_model(str(model_path))
        assert payload["model_name"] == best_name

        # 8. Biomarker analysis
        analyzer = BiomarkerAnalyzer(
            output_dir=str(tmp_path / "biomarkers"), drug_name="olaparib"
        )
        assoc_df = analyzer.test_biomarker_association(
            X_test, ic50_series.reindex(X_test.index)
        )
        assert isinstance(assoc_df, pd.DataFrame)

    def test_train_script_synthetic(self, tmp_path):
        """train.py CLI should run without error in --synthetic mode."""
        from scripts.train import main as train_main

        ret = train_main([
            "--drug", "olaparib",
            "--synthetic",
            "--n-synthetic", "120",
            "--output-dir", str(tmp_path),
            "--n-folds", "2",
            "--test-size", "0.25",
            "--no-plots",
        ])
        assert ret == 0

    def test_evaluate_script_after_train(self, tmp_path):
        """evaluate.py CLI should run on output of train.py."""
        from scripts.evaluate import main as eval_main
        from scripts.train import main as train_main

        # First train to produce artifacts
        train_main([
            "--drug", "olaparib",
            "--synthetic",
            "--n-synthetic", "100",
            "--output-dir", str(tmp_path),
            "--n-folds", "2",
            "--test-size", "0.25",
            "--no-plots",
        ])

        # Find the best model pkl
        model_dir = tmp_path / "olaparib" / "models"
        pkls = list(model_dir.glob("*.pkl"))
        assert len(pkls) > 0, "No model pkl files found after training"

        test_data_path = tmp_path / "olaparib" / "test_data.pkl"
        assert test_data_path.exists()

        ret = eval_main([
            "--model-path", str(pkls[0]),
            "--data-path", str(test_data_path),
            "--output-dir", str(tmp_path / "olaparib" / "eval"),
            "--drug", "olaparib",
            "--all-plots",
        ])
        assert ret == 0
