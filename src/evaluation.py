"""
Visualisation utilities for the DDR Biomarker Pipeline.

Generates ROC curves, precision-recall curves, confusion matrices,
and model-comparison bar charts.
"""

from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)

from .utils import ensure_dir, setup_logging

logger = setup_logging(__name__)


def plot_roc_curves(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for one or more models on the same axes.

    Args:
        results_dict: Maps model name to result dict with ``y_true`` and
            ``y_prob`` arrays.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in results_dict.items():
        y_true = result["y_true"]
        y_prob = result["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — DDR Biomarker Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        ensure_dir("/".join(save_path.replace("\\", "/").split("/")[:-1]))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_precision_recall(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot precision-recall curves for one or more models.

    Args:
        results_dict: Maps model name to result dict with ``y_true`` and
            ``y_prob`` arrays.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in results_dict.items():
        y_true = result["y_true"]
        y_prob = result["y_prob"]
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        ax.plot(recall_vals, precision_vals, lw=2,
                label=f"{name} (AP = {pr_auc:.3f})")

    baseline = np.mean(np.concatenate(
        [r["y_true"] for r in results_dict.values()]
    ))
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, label="Baseline")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — DDR Biomarker Models", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        ensure_dir("/".join(save_path.replace("\\", "/").split("/")[:-1]))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Precision-Recall curve saved to %s", save_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion-matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        save_path: If provided, save the figure to this path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Resistant", "Sensitive"],
        yticklabels=["Resistant", "Sensitive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        ensure_dir("/".join(save_path.replace("\\", "/").split("/")[:-1]))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


def plot_model_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing mean CV metrics across models.

    Args:
        results_dict: Maps model name to result dict with ``mean_metrics``.
        save_path: If provided, save the figure to this path.
    """
    rows = []
    for name, result in results_dict.items():
        row = {"Model": name}
        row.update(result["mean_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    metric_cols = [c for c in df.columns if c in ("auc", "accuracy", "precision", "recall", "f1")]
    df = df[metric_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_cols))
    width = 0.8 / len(df)
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, (model_name, values) in enumerate(df.iterrows()):
        offset = (i - len(df) / 2 + 0.5) * width
        ax.bar(x + offset, values.values, width, label=model_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in metric_cols], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        ensure_dir("/".join(save_path.replace("\\", "/").split("/")[:-1]))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison chart saved to %s", save_path)
    plt.close(fig)
