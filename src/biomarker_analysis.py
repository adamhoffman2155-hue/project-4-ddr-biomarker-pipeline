"""
Biomarker discovery analysis for the DDR Biomarker Pipeline.

Combines SHAP-based feature importance, univariate statistical tests, and
effect-size estimates into a single ranked biomarker summary table.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .utils import setup_logging

logger = setup_logging(__name__)


def run_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute SHAP values for a fitted model and rank features.

    Uses :class:`shap.TreeExplainer` for tree-based models and
    :class:`shap.LinearExplainer` for linear models, falling back to
    :class:`shap.KernelExplainer` when needed.

    Args:
        model: A fitted sklearn estimator.
        X: Feature matrix used for explanation.
        feature_names: Optional list of feature names; defaults to X.columns.

    Returns:
        DataFrame with columns ``feature``, ``mean_abs_shap``, ranked by
        importance (descending).
    """
    import shap

    if feature_names is None:
        feature_names = list(X.columns)

    model_name = type(model).__name__

    try:
        if hasattr(model, "estimators_") or "Boosting" in model_name or "Forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif hasattr(model, "coef_"):
            # Linear model — use a masker with a small background sample
            background = shap.maskers.Independent(X, max_samples=50)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        else:
            background = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X)
    except Exception as exc:
        logger.warning("SHAP analysis failed (%s), using coefficient fallback", exc)
        # Fallback: use model coefficients or feature importances
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(feature_names))

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return shap_df

    # shap_values may be a list (one array per class) or a single array
    if isinstance(shap_values, list):
        # Use the positive-class SHAP values
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("SHAP analysis complete — top feature: %s (%.4f)",
                shap_df.iloc[0]["feature"], shap_df.iloc[0]["mean_abs_shap"])

    return shap_df


def run_statistical_tests(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run Mann-Whitney U tests for each feature between classes.

    Applies Benjamini-Hochberg FDR correction to the raw p-values.

    Args:
        X: Feature matrix.
        y: Binary label vector (0/1).
        feature_names: Optional feature names; defaults to X.columns.

    Returns:
        DataFrame with columns ``feature``, ``u_statistic``, ``p_value``,
        ``p_adjusted`` (BH-corrected), sorted by adjusted p-value.
    """
    if feature_names is None:
        feature_names = list(X.columns)

    sensitive_mask = y == 1
    resistant_mask = y == 0

    records: List[Dict[str, Any]] = []
    for feat in feature_names:
        vals_sens = X.loc[sensitive_mask, feat].values
        vals_res = X.loc[resistant_mask, feat].values

        # Skip features with zero variance in both groups
        if np.std(vals_sens) == 0 and np.std(vals_res) == 0:
            records.append({
                "feature": feat,
                "u_statistic": 0.0,
                "p_value": 1.0,
            })
            continue

        try:
            u_stat, p_val = stats.mannwhitneyu(
                vals_sens, vals_res, alternative="two-sided"
            )
        except ValueError:
            u_stat, p_val = 0.0, 1.0

        records.append({
            "feature": feat,
            "u_statistic": float(u_stat),
            "p_value": float(p_val),
        })

    stats_df = pd.DataFrame(records)

    # Benjamini-Hochberg FDR correction
    stats_df = stats_df.sort_values("p_value").reset_index(drop=True)
    n_tests = len(stats_df)
    ranks = np.arange(1, n_tests + 1)
    raw_p = stats_df["p_value"].values

    # BH adjusted p-values
    bh_adjusted = np.minimum(1.0, raw_p * n_tests / ranks)
    # Enforce monotonicity (from largest rank to smallest)
    bh_adjusted_mono = np.empty_like(bh_adjusted)
    bh_adjusted_mono[-1] = bh_adjusted[-1]
    for i in range(n_tests - 2, -1, -1):
        bh_adjusted_mono[i] = min(bh_adjusted[i], bh_adjusted_mono[i + 1])

    stats_df["p_adjusted"] = bh_adjusted_mono

    logger.info(
        "Statistical tests complete — %d features, %d significant (FDR<0.05)",
        n_tests, int((stats_df["p_adjusted"] < 0.05).sum()),
    )

    return stats_df


def compute_effect_size(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Cohen's d effect size for each feature between classes.

    Args:
        X: Feature matrix.
        y: Binary label vector.
        feature_names: Optional feature names; defaults to X.columns.

    Returns:
        DataFrame with columns ``feature``, ``cohens_d``, ``mean_sensitive``,
        ``mean_resistant``, sorted by absolute effect size (descending).
    """
    if feature_names is None:
        feature_names = list(X.columns)

    sensitive_mask = y == 1
    resistant_mask = y == 0

    records: List[Dict[str, float]] = []
    for feat in feature_names:
        vals_sens = X.loc[sensitive_mask, feat].values.astype(float)
        vals_res = X.loc[resistant_mask, feat].values.astype(float)

        mean_s = float(np.mean(vals_sens))
        mean_r = float(np.mean(vals_res))
        std_s = float(np.std(vals_sens, ddof=1)) if len(vals_sens) > 1 else 0.0
        std_r = float(np.std(vals_res, ddof=1)) if len(vals_res) > 1 else 0.0

        # Pooled standard deviation
        n_s, n_r = len(vals_sens), len(vals_res)
        if n_s + n_r - 2 > 0 and (std_s > 0 or std_r > 0):
            pooled_std = np.sqrt(
                ((n_s - 1) * std_s ** 2 + (n_r - 1) * std_r ** 2)
                / (n_s + n_r - 2)
            )
            cohens_d = (mean_s - mean_r) / pooled_std if pooled_std > 0 else 0.0
        else:
            cohens_d = 0.0

        records.append({
            "feature": feat,
            "cohens_d": float(cohens_d),
            "abs_cohens_d": float(abs(cohens_d)),
            "mean_sensitive": mean_s,
            "mean_resistant": mean_r,
        })

    effect_df = pd.DataFrame(records).sort_values(
        "abs_cohens_d", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Effect size analysis complete — top feature: %s (d=%.3f)",
        effect_df.iloc[0]["feature"], effect_df.iloc[0]["cohens_d"],
    )

    return effect_df


def summarize_biomarkers(
    shap_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    effect_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge SHAP, statistical, and effect-size results into a ranked summary.

    Features are ranked by a composite score:
        score = rank(mean_abs_shap) + rank(1/p_adjusted) + rank(abs_cohens_d)

    Lower composite rank = stronger biomarker evidence.

    Args:
        shap_df: Output of :func:`run_shap_analysis`.
        stats_df: Output of :func:`run_statistical_tests`.
        effect_df: Output of :func:`compute_effect_size`.

    Returns:
        Merged DataFrame sorted by composite rank, with all analysis columns.
    """
    merged = shap_df.merge(stats_df, on="feature", how="outer")
    merged = merged.merge(
        effect_df[["feature", "cohens_d", "abs_cohens_d"]],
        on="feature",
        how="outer",
    )

    # Fill NaN for features missing from any analysis
    merged["mean_abs_shap"] = merged["mean_abs_shap"].fillna(0)
    merged["p_adjusted"] = merged["p_adjusted"].fillna(1)
    merged["abs_cohens_d"] = merged["abs_cohens_d"].fillna(0)

    # Rank each metric (higher = better biomarker)
    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False)
    merged["pval_rank"] = merged["p_adjusted"].rank(ascending=True)
    merged["effect_rank"] = merged["abs_cohens_d"].rank(ascending=False)

    # Composite score (lower = better)
    merged["composite_rank"] = merged["shap_rank"] + merged["pval_rank"] + merged["effect_rank"]
    merged = merged.sort_values("composite_rank").reset_index(drop=True)

    logger.info("Biomarker summary — top 5:")
    for i, row in merged.head(5).iterrows():
        logger.info(
            "  %d. %s  SHAP=%.4f  p_adj=%.4f  d=%.3f",
            i + 1, row["feature"], row["mean_abs_shap"],
            row["p_adjusted"], row.get("cohens_d", 0),
        )

    return merged
