"""
Model training module for the DDR Biomarker Pipeline.

Implements ModelTrainer which trains, cross-validates, and serializes
four complementary ML models:

  - Logistic Regression (interpretable linear baseline)
  - Random Forest (non-linear, handles feature interactions)
  - Gradient Boosting (best single-model performance)
  - Elastic Net (regularized linear, feature selection)

All models are trained with stratified 5-fold cross-validation and
evaluated on AUC-ROC, AUC-PR, F1, and accuracy.

Typical usage::

    from src.models import ModelTrainer
    from config.config import PipelineConfig

    cfg = PipelineConfig()
    trainer = ModelTrainer(cfg)
    results = trainer.train_all_models(X_train, y_train)
    best = trainer.get_best_model()
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.config import PipelineConfig

logger = logging.getLogger(__name__)


# Type alias
MetricsDict = Dict[str, float]


class ModelTrainer:
    """
    Trains and evaluates ML models for DDR drug sensitivity prediction.

    Parameters
    ----------
    config : PipelineConfig
    """

    # Model registry: name -> constructor
    MODEL_NAMES = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "elastic_net",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._models: Dict[str, Any] = {}
        self._cv_results: Dict[str, Dict] = {}
        self._best_model_name: Optional[str] = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Model constructors
    # ------------------------------------------------------------------

    def _build_logistic_regression(self) -> Pipeline:
        """Build logistic regression pipeline with StandardScaler."""
        params = self.config.model.logistic_regression.copy()
        clf = LogisticRegression(**params)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _build_random_forest(self) -> RandomForestClassifier:
        """Build random forest classifier."""
        params = self.config.model.random_forest.copy()
        return RandomForestClassifier(**params)

    def _build_gradient_boosting(self) -> GradientBoostingClassifier:
        """Build gradient boosting classifier."""
        params = self.config.model.gradient_boosting.copy()
        return GradientBoostingClassifier(**params)

    def _build_elastic_net(self) -> Pipeline:
        """
        Build Elastic Net classifier using SGDClassifier with elasticnet penalty.
        SGDClassifier supports elasticnet and outputs probabilities via calibration.
        """
        params = self.config.model.elastic_net.copy()
        alpha = params.get("alpha", 0.01)
        l1_ratio = params.get("l1_ratio", 0.5)
        random_state = params.get("random_state", 42)
        class_weight = params.get("class_weight", "balanced")

        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=params.get("max_iter", 2000),
            tol=params.get("tol", 1e-4),
            class_weight=class_weight,
            random_state=random_state,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _build_all_models(self) -> Dict[str, Any]:
        """Instantiate all four models."""
        return {
            "logistic_regression": self._build_logistic_regression(),
            "random_forest": self._build_random_forest(),
            "gradient_boosting": self._build_gradient_boosting(),
            "elastic_net": self._build_elastic_net(),
        }

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run stratified k-fold cross-validation for a single model.

        Parameters
        ----------
        model : sklearn estimator
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary labels (0/1).
        n_folds : int, optional
            Number of folds. Defaults to config value.

        Returns
        -------
        dict with keys:
            mean_roc_auc, std_roc_auc,
            mean_pr_auc, std_pr_auc,
            mean_f1, std_f1,
            mean_accuracy, std_accuracy,
            fold_scores (list of per-fold metric dicts)
        """
        if n_folds is None:
            n_folds = self.config.cv.n_folds

        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=self.config.cv.shuffle,
            random_state=self.config.cv.random_state,
        )

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            model.fit(X_train, y_train)

            # Predict
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.decision_function(X_val)
                # Normalize to [0,1]
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

            y_pred = model.predict(X_val)

            # Check that both classes present in fold
            if len(np.unique(y_val)) < 2:
                logger.warning("Fold %d has only one class; skipping AUC.", fold_idx)
                continue

            fold_metrics = {
                "fold": fold_idx,
                "roc_auc": roc_auc_score(y_val, y_prob),
                "pr_auc": average_precision_score(y_val, y_prob),
                "f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
                "accuracy": accuracy_score(y_val, y_pred),
                "n_val": len(y_val),
                "n_positive_val": int(y_val.sum()),
            }
            fold_scores.append(fold_metrics)
            logger.debug(
                "  Fold %d — ROC-AUC: %.3f | PR-AUC: %.3f | F1: %.3f",
                fold_idx, fold_metrics["roc_auc"], fold_metrics["pr_auc"],
                fold_metrics["f1"],
            )

        if not fold_scores:
            logger.error("No valid folds produced metrics.")
            return {
                "mean_roc_auc": 0.5, "std_roc_auc": 0.0,
                "mean_pr_auc": 0.5, "std_pr_auc": 0.0,
                "mean_f1": 0.0, "std_f1": 0.0,
                "mean_accuracy": 0.5, "std_accuracy": 0.0,
                "fold_scores": [],
            }

        roc_aucs = [s["roc_auc"] for s in fold_scores]
        pr_aucs = [s["pr_auc"] for s in fold_scores]
        f1s = [s["f1"] for s in fold_scores]
        accs = [s["accuracy"] for s in fold_scores]

        return {
            "mean_roc_auc": float(np.mean(roc_aucs)),
            "std_roc_auc": float(np.std(roc_aucs)),
            "mean_pr_auc": float(np.mean(pr_aucs)),
            "std_pr_auc": float(np.std(pr_aucs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "fold_scores": fold_scores,
        }

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train all four models with cross-validation and return results.

        After calling this, use get_best_model() to retrieve the top-
        performing model, or access self._models for all fitted models.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        n_folds : int, optional
        verbose : bool

        Returns
        -------
        dict mapping model_name -> cv_metrics_dict
        """
        self._feature_names = list(X_train.columns)
        models = self._build_all_models()

        if verbose:
            logger.info("=" * 60)
            logger.info("Training %d models with %d-fold CV",
                        len(models), n_folds or self.config.cv.n_folds)
            logger.info("Training set: %d samples, %d features",
                        X_train.shape[0], X_train.shape[1])
            logger.info("Class balance: %d sensitive / %d resistant",
                        y_train.sum(), (y_train == 0).sum())
            logger.info("=" * 60)

        all_results: Dict[str, Dict] = {}

        for model_name, model in models.items():
            logger.info("Training %s ...", model_name)
            cv_results = self.cross_validate(model, X_train, y_train, n_folds)
            all_results[model_name] = cv_results

            if verbose:
                logger.info(
                    "  %s: ROC-AUC = %.3f ± %.3f | PR-AUC = %.3f ± %.3f | "
                    "F1 = %.3f ± %.3f",
                    model_name,
                    cv_results["mean_roc_auc"], cv_results["std_roc_auc"],
                    cv_results["mean_pr_auc"], cv_results["std_pr_auc"],
                    cv_results["mean_f1"], cv_results["std_f1"],
                )

            # Refit on full training set
            model.fit(
                X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                y_train.values if isinstance(y_train, pd.Series) else y_train,
            )
            self._models[model_name] = model
            self._cv_results[model_name] = cv_results

        # Select best model by mean CV ROC-AUC
        self._best_model_name = max(
            all_results,
            key=lambda k: all_results[k]["mean_roc_auc"],
        )
        logger.info(
            "Best model: %s (ROC-AUC = %.3f)",
            self._best_model_name,
            all_results[self._best_model_name]["mean_roc_auc"],
        )

        return all_results

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    def get_best_model(self) -> Tuple[str, Any]:
        """
        Return the name and fitted instance of the best-performing model.

        Returns
        -------
        Tuple[str, estimator]
        """
        if self._best_model_name is None:
            raise RuntimeError("No models trained yet. Call train_all_models() first.")
        return self._best_model_name, self._models[self._best_model_name]

    def get_model(self, name: str) -> Any:
        """Return a specific fitted model by name."""
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def get_cv_results_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of CV results across all models.

        Returns
        -------
        pd.DataFrame
            Rows = models, columns = metrics.
        """
        if not self._cv_results:
            raise RuntimeError("No CV results available. Call train_all_models() first.")

        rows = []
        for model_name, results in self._cv_results.items():
            rows.append({
                "model": model_name,
                "roc_auc_mean": results["mean_roc_auc"],
                "roc_auc_std": results["std_roc_auc"],
                "pr_auc_mean": results["mean_pr_auc"],
                "pr_auc_std": results["std_pr_auc"],
                "f1_mean": results["mean_f1"],
                "f1_std": results["std_f1"],
                "accuracy_mean": results["mean_accuracy"],
                "accuracy_std": results["std_accuracy"],
            })

        return pd.DataFrame(rows).set_index("model")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_proba(
        self, model_name: str, X: pd.DataFrame
    ) -> np.ndarray:
        """
        Return predicted probabilities for class 1 (sensitive).

        Parameters
        ----------
        model_name : str
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        model = self.get_model(model_name)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_arr)[:, 1]
        else:
            scores = model.decision_function(X_arr)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0/1) for a given model."""
        model = self.get_model(model_name)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return model.predict(X_arr)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_model(
        self,
        model_name: str,
        output_dir: str,
        include_metadata: bool = True,
    ) -> Path:
        """
        Serialize a trained model to disk using pickle.

        Parameters
        ----------
        model_name : str
        output_dir : str
        include_metadata : bool
            If True, save a companion JSON with CV metrics.

        Returns
        -------
        Path
            Path to the saved .pkl file.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not trained yet.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / f"{model_name}_model.pkl"
        payload = {
            "model": self._models[model_name],
            "model_name": model_name,
            "feature_names": self._feature_names,
            "cv_results": self._cv_results.get(model_name, {}),
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Model saved: %s", model_path)

        if include_metadata and model_name in self._cv_results:
            import json
            meta_path = out_dir / f"{model_name}_cv_metrics.json"
            # Convert numpy types for JSON serialization
            cv = self._cv_results[model_name].copy()
            cv.pop("fold_scores", None)  # Remove verbose fold-level data
            with open(meta_path, "w") as f:
                json.dump({k: round(float(v), 4) for k, v in cv.items()}, f, indent=2)

        return model_path

    def save_all_models(self, output_dir: str) -> List[Path]:
        """Save all trained models to disk."""
        paths = []
        for model_name in self._models:
            path = self.save_model(model_name, output_dir)
            paths.append(path)
        return paths

    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        Load a serialized model from disk.

        Parameters
        ----------
        model_path : str
            Path to .pkl file saved by save_model().

        Returns
        -------
        dict with keys: model, model_name, feature_names, cv_results.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        logger.info(
            "Loaded model '%s' from %s", payload.get("model_name"), model_path
        )
        return payload
