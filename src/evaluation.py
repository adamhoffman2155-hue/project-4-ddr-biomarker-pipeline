"""
Evaluation module for the DDR Biomarker Pipeline.

Provides ModelEvaluator with methods to:
  - Compute AUC-ROC, AUC-PR, F1, accuracy on held-out test sets
  - Plot ROC curves and Precision-Recall curves (matplotlib)
  - Plot confusion matrices
  - Plot feature importance (coefficient-based or Gini impurity)
  - Generate classification reports
  - Compare multiple models in a summary DataFrame

Typical usage::

    from src.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(output_dir="results/olaparib/evaluation")
    evaluator.compute_roc_auc(y_true, y_prob)
    evaluator.plot_roc_curves(all_y_probs_dict, y_true)
    evaluator.generate_report(y_true, y_pred, model_name="gradient_boosting")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Consistent color palette
PALETTE = {
    "logistic_regression": "#2196F3",   # Blue
    "random_forest": "#4CAF50",         # Green
    "gradient_boosting": "#F44336",     # Red
    "elastic_net": "#FF9800",           # Orange
}
DEFAULT_COLOR = "#9C27B0"              # Purple fallback


class ModelEvaluator:
    """
    Evaluates trained DDR biomarker models on held-out test data.

    Parameters
    ----------
    output_dir : str
        Directory to save plots and reports.
    drug_name : str
        Drug name for plot titles.
    dpi : int
        DPI for saved figures.
    """

    def __init__(
        self,
        output_dir: str = "results/evaluation",
        drug_name: str = "",
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drug_name = drug_name
        self.dpi = dpi
        self._results_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Scalar metrics
    # ------------------------------------------------------------------

    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> float:
        """
        Compute AUC-ROC score.

        Parameters
        ----------
        y_true : array-like of shape (n,)
            Binary ground-truth labels.
        y_prob : array-like of shape (n,)
            Predicted probabilities for class 1.
        model_name : str
            For logging.

        Returns
        -------
        float
        """
        auc = roc_auc_score(y_true, y_prob)
        logger.info("%s ROC-AUC: %.4f", model_name, auc)
        return float(auc)

    def compute_pr_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> float:
        """
        Compute AUC-PR (average precision) score.

        Parameters
        ----------
        y_true : array-like
        y_prob : array-like
        model_name : str

        Returns
        -------
        float
        """
        auc_pr = average_precision_score(y_true, y_prob)
        logger.info("%s AUC-PR: %.4f", model_name, auc_pr)
        return float(auc_pr)

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        model_name: str = "model",
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute a full metrics dictionary for a model.

        Parameters
        ----------
        y_true, y_prob, y_pred : array-like
        model_name : str
        threshold : float
            Probability threshold for converting y_prob to binary predictions.

        Returns
        -------
        dict with keys: roc_auc, pr_auc, f1, accuracy, kappa
        """
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = np.asarray(y_pred)

        metrics = {
            "model": model_name,
            "roc_auc": self.compute_roc_auc(y_true, y_prob, model_name),
            "pr_auc": self.compute_pr_auc(y_true, y_prob, model_name),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
        }
        self._results_cache[model_name] = metrics
        return metrics

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        model_probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        title: Optional[str] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models on a single axes.

        Parameters
        ----------
        model_probs : dict mapping model_name -> y_prob array
        y_true : array-like
        title : str, optional
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        y_true = np.asarray(y_true)
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, y_prob in model_probs.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            color = PALETTE.get(model_name, DEFAULT_COLOR)
            label = f"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})"
            ax.plot(fpr, tpr, lw=2, color=color, label=label)

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "--", color="#BDBDBD", lw=1.5, label="Random (AUC = 0.500)")

        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        title_str = title or f"ROC Curves — {self.drug_name or 'Drug'} Sensitivity"
        ax.set_title(title_str, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = self.output_dir / "roc_curves.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("ROC curves saved: %s", path)

        return fig

    def plot_pr_curves(
        self,
        model_probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        title: Optional[str] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.

        Parameters
        ----------
        model_probs : dict
        y_true : array-like
        title : str, optional
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        y_true = np.asarray(y_true)
        baseline = y_true.mean()

        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, y_prob in model_probs.items():
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            color = PALETTE.get(model_name, DEFAULT_COLOR)
            label = f"{model_name.replace('_', ' ').title()} (AP = {ap:.3f})"
            ax.plot(recall, precision, lw=2, color=color, label=label)

        # Baseline (random classifier)
        ax.axhline(y=baseline, linestyle="--", color="#BDBDBD", lw=1.5,
                   label=f"Baseline (AP = {baseline:.3f})")

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        title_str = title or f"Precision-Recall Curves — {self.drug_name or 'Drug'}"
        ax.set_title(title_str, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = self.output_dir / "pr_curves.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("PR curves saved: %s", path)

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        class_names: Optional[List[str]] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot a normalized confusion matrix heatmap.

        Parameters
        ----------
        y_true, y_pred : array-like
        model_name : str
        class_names : list of str, optional
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        if class_names is None:
            class_names = ["Resistant (0)", "Sensitive (1)"]

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {model_name.replace('_', ' ').title()}",
            fontsize=13, fontweight="bold",
        )

        # Add raw counts as secondary annotation
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.73, f"n={cm[i,j]}",
                    ha="center", va="center", fontsize=9, color="gray",
                )

        fig.tight_layout()

        if save:
            fname = f"confusion_matrix_{model_name}.png"
            path = self.output_dir / fname
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Confusion matrix saved: %s", path)

        return fig

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "model",
        top_n: int = 20,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot feature importances from a trained model.

        Uses feature_importances_ (RF/GB) or coef_ (LR/EN).

        Parameters
        ----------
        model : sklearn estimator or Pipeline
        feature_names : list of str
        model_name : str
        top_n : int
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        # Extract importance scores
        importances = self._extract_feature_importances(model, len(feature_names))
        if importances is None:
            logger.warning("Cannot extract importances from %s", model_name)
            return plt.figure()

        df_imp = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": np.abs(importances),
        }).sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        colors = [PALETTE.get(model_name, DEFAULT_COLOR)] * len(df_imp)
        ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color=colors)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features — {model_name.replace('_', ' ').title()}",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if save:
            path = self.output_dir / f"feature_importance_{model_name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Feature importance plot saved: %s", path)

        return fig

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        save: bool = True,
    ) -> str:
        """
        Generate and optionally save a full classification report.

        Parameters
        ----------
        y_true, y_pred : array-like
        model_name : str
        save : bool

        Returns
        -------
        str
            The classification report string.
        """
        report = classification_report(
            y_true, y_pred,
            target_names=["Resistant", "Sensitive"],
            zero_division=0,
        )
        header = (
            f"\n{'='*60}\n"
            f"Classification Report — {model_name.replace('_', ' ').title()}\n"
            f"Drug: {self.drug_name or 'N/A'}\n"
            f"{'='*60}\n"
        )
        full_report = header + report + "\n"
        print(full_report)

        if save:
            path = self.output_dir / f"classification_report_{model_name}.txt"
            path.write_text(full_report)
            logger.info("Classification report saved: %s", path)

        return full_report

    def compare_models(
        self,
        model_probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        y_preds: Optional[Dict[str, np.ndarray]] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Compare all models in a summary metrics DataFrame.

        Parameters
        ----------
        model_probs : dict mapping model_name -> y_prob
        y_true : array-like
        y_preds : dict mapping model_name -> y_pred (binary), optional
        save : bool

        Returns
        -------
        pd.DataFrame sorted by ROC-AUC descending.
        """
        y_true = np.asarray(y_true)
        rows = []

        for model_name, y_prob in model_probs.items():
            y_pred = y_preds.get(model_name) if y_preds else (y_prob >= 0.5).astype(int)
            metrics = self.compute_all_metrics(
                y_true, y_prob, y_pred, model_name
            )
            rows.append(metrics)

        df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

        if save:
            path = self.output_dir / "model_comparison.csv"
            df.to_csv(path, index=False)
            logger.info("Model comparison saved: %s", path)

        return df

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot a grouped bar chart comparing models across key metrics.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output of compare_models().
        metrics : list of str, optional
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        if metrics is None:
            metrics = ["roc_auc", "pr_auc", "f1", "accuracy"]

        df = comparison_df.set_index("model")[metrics]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        n_metrics = len(metrics)
        bar_width = 0.8 / n_metrics
        offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_metrics)

        metric_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
        for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
            ax.bar(
                x + offsets[i], df[metric], bar_width,
                label=metric.replace("_", " ").upper(),
                color=color, alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in df.index], fontsize=11
        )
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        title_str = f"Model Comparison — {self.drug_name or 'Drug'} Sensitivity"
        ax.set_title(title_str, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save:
            path = self.output_dir / "model_comparison.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Model comparison plot saved: %s", path)

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_feature_importances(
        self, model: Any, n_features: int
    ) -> Optional[np.ndarray]:
        """Extract feature importance vector from a fitted model."""
        # Unwrap Pipeline
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("clf", model)
        else:
            estimator = model

        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return np.abs(coef)
        return None
