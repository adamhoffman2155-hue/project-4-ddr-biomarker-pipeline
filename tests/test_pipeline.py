"""
Tests for the DDR Biomarker Pipeline.

Run with::

    cd project-4-ddr-biomarker-pipeline
    pytest tests/test_pipeline.py -v
"""

import os
import sys

import pytest

# Ensure the project root is importable
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
from src.feature_engineering import (
    build_feature_matrix,
    compute_ddr_burden,
    compute_hrd_score,
    compute_msi_status,
)
from src.models import (
    evaluate_model,
    train_gradient_boosting,
    train_logistic_regression,
)


@pytest.fixture
def config():
    """Return a default PipelineConfig."""
    return PipelineConfig()


@pytest.fixture
def synthetic_data(config):
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_data(
        n_cell_lines=100,
        seed=config.RANDOM_SEED,
        ddr_genes=config.DDR_GENES,
        drug_names=config.DDR_DRUGS,
        mutation_rate=config.MUTATION_RATE,
    )


@pytest.fixture
def feature_data(synthetic_data, config):
    """Build feature matrix for olaparib."""
    data = synthetic_data
    X, y = build_feature_matrix(data["mutations"], data["ic50"], "olaparib", config)
    return X, y


# ------------------------------------------------------------------
# Data loader tests
# ------------------------------------------------------------------


class TestSyntheticData:
    def test_synthetic_data_shape(self, synthetic_data, config):
        """IC50, mutations, and metadata all have 100 cell lines."""
        ic50 = synthetic_data["ic50"]
        mut = synthetic_data["mutations"]
        meta = synthetic_data["metadata"]

        assert ic50.shape == (100, len(config.DDR_DRUGS))
        assert mut.shape == (100, len(config.DDR_GENES))
        assert meta.shape[0] == 100

    def test_mutation_matrix_is_binary(self, synthetic_data):
        """Mutation matrix should contain only 0s and 1s."""
        mut = synthetic_data["mutations"]
        assert set(mut.values.flatten()).issubset({0, 1})

    def test_merge_preserves_rows(self, synthetic_data):
        """Merging IC50 and mutation matrices should keep all shared rows."""
        merged = merge_datasets(synthetic_data["ic50"], synthetic_data["mutations"])
        assert len(merged) == 100


# ------------------------------------------------------------------
# Feature engineering tests
# ------------------------------------------------------------------


class TestFeatureEngineering:
    def test_feature_engineering_output(self, feature_data, config):
        """X should have DDR gene cols plus 3 composite features."""
        X, y = feature_data
        expected_cols = len(config.DDR_GENES) + 3  # hrd, msi, burden
        assert X.shape[1] == expected_cols
        assert len(y) == len(X)

    def test_hrd_score_range(self, synthetic_data, config):
        """HRD score must be in [0, len(HR_GENES)]."""
        mut = synthetic_data["mutations"]
        for _, row in mut.iterrows():
            score = compute_hrd_score(row, config.HR_GENES)
            assert 0 <= score <= len(config.HR_GENES)

    def test_msi_status_binary(self, synthetic_data):
        """MSI status must be 0 or 1."""
        mut = synthetic_data["mutations"]
        for _, row in mut.iterrows():
            status = compute_msi_status(row)
            assert status in (0, 1)

    def test_ddr_burden_range(self, synthetic_data, config):
        """DDR burden must be in [0, len(DDR_GENES)]."""
        mut = synthetic_data["mutations"]
        for _, row in mut.iterrows():
            burden = compute_ddr_burden(row, config.DDR_GENES)
            assert 0 <= burden <= len(config.DDR_GENES)

    def test_label_is_binary(self, feature_data):
        """y should only contain 0 and 1."""
        _, y = feature_data
        assert set(y.unique()).issubset({0, 1})


# ------------------------------------------------------------------
# Model tests
# ------------------------------------------------------------------


class TestModels:
    def test_model_training_returns_metrics(self, feature_data, config):
        """Both training functions should return a dict with required keys."""
        X, y = feature_data

        lr_result = train_logistic_regression(X, y, config)
        assert "model" in lr_result
        assert "mean_metrics" in lr_result
        assert "auc" in lr_result["mean_metrics"]
        assert 0 <= lr_result["mean_metrics"]["auc"] <= 1

        gb_result = train_gradient_boosting(X, y, config)
        assert "model" in gb_result
        assert "mean_metrics" in gb_result
        assert 0 <= gb_result["mean_metrics"]["auc"] <= 1

    def test_evaluate_model_keys(self, feature_data, config):
        """evaluate_model should return all expected metric keys."""
        X, y = feature_data
        lr_result = train_logistic_regression(X, y, config)
        model = lr_result["model"]

        metrics = evaluate_model(model, X, y)
        for key in ("auc", "accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0 <= metrics[key] <= 1


# ------------------------------------------------------------------
# Biomarker analysis tests
# ------------------------------------------------------------------


class TestBiomarkerAnalysis:
    def test_shap_values_shape(self, feature_data, config):
        """SHAP analysis should return one row per feature."""
        X, y = feature_data
        gb_result = train_gradient_boosting(X, y, config)
        model = gb_result["model"]

        shap_df = run_shap_analysis(model, X, list(X.columns))
        assert len(shap_df) == X.shape[1]
        assert "feature" in shap_df.columns
        assert "mean_abs_shap" in shap_df.columns

    def test_statistical_tests_output(self, feature_data):
        """Statistical tests should return p-values and adjusted p-values."""
        X, y = feature_data
        stats_df = run_statistical_tests(X, y, list(X.columns))

        assert len(stats_df) == X.shape[1]
        assert "p_value" in stats_df.columns
        assert "p_adjusted" in stats_df.columns
        assert (stats_df["p_value"] >= 0).all()
        assert (stats_df["p_adjusted"] >= 0).all()
        assert (stats_df["p_adjusted"] <= 1).all()

    def test_effect_size_output(self, feature_data):
        """Effect size should return Cohen's d for each feature."""
        X, y = feature_data
        effect_df = compute_effect_size(X, y, list(X.columns))

        assert len(effect_df) == X.shape[1]
        assert "cohens_d" in effect_df.columns
        assert "abs_cohens_d" in effect_df.columns

    def test_summarize_biomarkers(self, feature_data, config):
        """Biomarker summary should merge all analysis results."""
        X, y = feature_data
        gb_result = train_gradient_boosting(X, y, config)
        model = gb_result["model"]
        feature_names = list(X.columns)

        shap_df = run_shap_analysis(model, X, feature_names)
        stats_df = run_statistical_tests(X, y, feature_names)
        effect_df = compute_effect_size(X, y, feature_names)
        summary = summarize_biomarkers(shap_df, stats_df, effect_df)

        assert "composite_rank" in summary.columns
        assert len(summary) == X.shape[1]
