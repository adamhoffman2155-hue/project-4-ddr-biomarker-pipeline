"""
Model training and evaluation for the DDR Biomarker Pipeline.

Provides stratified cross-validated training for Logistic Regression and
Gradient Boosting classifiers, plus evaluation utilities for comparing
model performance.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

from .utils import setup_logging, set_seed

logger = setup_logging(__name__)


def _select_best_lr(
    X: pd.DataFrame,
    y: pd.Series,
    config: Any,
) -> Tuple[LogisticRegression, float]:
    """Select the best regularisation strength via inner CV.

    Args:
        X: Feature matrix.
        y: Binary label vector.
        config: Pipeline configuration with MODEL_PARAMS and RANDOM_SEED.

    Returns:
        Tuple of (best LogisticRegression instance, best mean AUC).
    """
    lr_params = config.MODEL_PARAMS["LogisticRegression"]
    best_auc = -1.0
    best_model: Optional[LogisticRegression] = None

    for c_val in lr_params["C_values"]:
        model = LogisticRegression(
            C=c_val,
            penalty=lr_params["penalty"],
            solver=lr_params["solver"],
            max_iter=lr_params["max_iter"],
            class_weight=lr_params["class_weight"],
            random_state=config.RANDOM_SEED,
        )
        inner_cv = StratifiedKFold(
            n_splits=min(3, config.N_FOLDS),
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )
        fold_aucs: List[float] = []
        for train_idx, val_idx in inner_cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            if len(np.unique(y_val)) < 2:
                continue
            proba = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, proba))

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model = LogisticRegression(
                C=c_val,
                penalty=lr_params["penalty"],
                solver=lr_params["solver"],
                max_iter=lr_params["max_iter"],
                class_weight=lr_params["class_weight"],
                random_state=config.RANDOM_SEED,
            )

    logger.info("Best LR C value selected (inner CV AUC=%.3f)", best_auc)
    return best_model, best_auc  # type: ignore[return-value]


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    config: Any,
) -> Dict[str, Any]:
    """Train a Logistic Regression model with stratified K-fold CV.

    The best regularisation strength is chosen by an inner CV loop, then
    the outer loop estimates generalisation performance.

    Args:
        X: Feature matrix (n_samples x n_features).
        y: Binary label vector.
        config: Pipeline configuration.

    Returns:
        Dictionary with keys ``"model"``, ``"cv_metrics"``,
        ``"mean_metrics"``, ``"y_true"``, ``"y_prob"``.
    """
    set_seed(config.RANDOM_SEED)
    logger.info("Training Logistic Regression (%d folds)", config.N_FOLDS)

    best_model, _ = _select_best_lr(X, y, config)

    cv = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_metrics: List[Dict[str, float]] = []
    all_y_true: List[np.ndarray] = []
    all_y_prob: List[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_model.fit(X_train, y_train)
        metrics = evaluate_model(best_model, X_test, y_test)
        fold_metrics.append(metrics)
        all_y_true.append(y_test.values)
        all_y_prob.append(best_model.predict_proba(X_test)[:, 1])

        logger.info("  Fold %d \u2014 AUC=%.3f  F1=%.3f", fold, metrics["auc"], metrics["f1"])

    # Refit on full data for downstream use
    best_model.fit(X, y)

    mean_metrics = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }
    logger.info(
        "LR mean CV \u2014 AUC=%.3f  Acc=%.3f  F1=%.3f",
        mean_metrics["auc"], mean_metrics["accuracy"], mean_metrics["f1"],
    )

    return {
        "model": best_model,
        "cv_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "y_true": np.concatenate(all_y_true),
        "y_prob": np.concatenate(all_y_prob),
    }


def train_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    config: Any,
) -> Dict[str, Any]:
    """Train a Gradient Boosting classifier with stratified K-fold CV.

    Args:
        X: Feature matrix (n_samples x n_features).
        y: Binary label vector.
        config: Pipeline configuration.

    Returns:
        Dictionary with keys ``"model"``, ``"cv_metrics"``,
        ``"mean_metrics"``, ``"y_true"``, ``"y_prob"``.
    """
    set_seed(config.RANDOM_SEED)
    logger.info("Training Gradient Boosting (%d folds)", config.N_FOLDS)

    gb_params = config.get_gb_params()

    cv = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_metrics: List[Dict[str, float]] = []
    all_y_true: List[np.ndarray] = []
    all_y_prob: List[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = GradientBoostingClassifier(
            random_state=config.RANDOM_SEED, **gb_params
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        fold_metrics.append(metrics)
        all_y_true.append(y_test.values)
        all_y_prob.append(model.predict_proba(X_test)[:, 1])

        logger.info("  Fold %d \u2014 AUC=%.3f  F1=%.3f", fold, metrics["auc"], metrics["f1"])

    # Refit on full data
    final_model = GradientBoostingClassifier(
        random_state=config.RANDOM_SEED, **gb_params
    )
    final_model.fit(X, y)

    mean_metrics = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }
    logger.info(
        "GBM mean CV \u2014 AUC=%.3f  Acc=%.3f  F1=%.3f",
        mean_metrics["auc"], mean_metrics["accuracy"], mean_metrics["f1"],
    )

    return {
        "model": final_model,
        "cv_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "y_true": np.concatenate(all_y_true),
        "y_prob": np.concatenate(all_y_prob),
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate a fitted classifier on held-out data.

    Args:
        model: A fitted sklearn estimator with ``predict`` and ``predict_proba``.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Dictionary with AUC, accuracy, precision, recall, and F1.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Handle edge case where only one class is present in test fold
    if len(np.unique(y_test)) < 2:
        auc_val = 0.5
    else:
        auc_val = roc_auc_score(y_test, y_prob)

    return {
        "auc": float(auc_val),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Print and return a comparison table of model performance.

    Args:
        results_dict: Maps model name to its training result dictionary
            (as returned by ``train_logistic_regression`` or
            ``train_gradient_boosting``).

    Returns:
        A :class:`~pandas.DataFrame` with one row per model and columns
        for each metric.
    """
    rows = []
    for name, result in results_dict.items():
        row = {"model": name}
        row.update(result["mean_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df.round(4)

    logger.info("\n=== Model Comparison ===\n%s", df.to_string())
    print("\n=== Model Comparison ===")
    print(df.to_string())
    print()

    return df
