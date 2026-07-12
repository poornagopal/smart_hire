"""
src/evaluate.py
-----------------
Shared evaluation helpers used across Phase 1 (classifier) and later phases
(fit predictor in Phase 3, clustering quality in Phase 2).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred, average="weighted") -> dict:
    """Returns a dict of accuracy/precision/recall/F1 — the core metrics for Model A & B."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def binary_roc_auc(y_true, y_proba) -> float:
    """ROC-AUC for the (optional) binary shortlisting / fit predictor, Phase 3."""
    return roc_auc_score(y_true, y_proba)


def print_classification_report(y_true, y_pred, target_names=None) -> str:
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    logger.info("\n%s", report)
    return report


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(10, 8), save_path=None):
    """
    Plots (and optionally saves) a confusion matrix heatmap.
    For 25-class resume categories, figsize is intentionally large for readability.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=labels is not None and len(labels) <= 15,  # avoid clutter for 25 classes
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved confusion matrix -> %s", save_path)

    return fig


def compare_models(results: dict) -> None:
    """
    results: {"logistic_regression": {"accuracy": .., "f1": ..}, "random_forest": {...}, ...}
    Logs a simple ranked comparison table by F1 score.
    """
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("f1", 0), reverse=True)
    logger.info("Model comparison (ranked by F1):")
    for name, metrics in ranked:
        logger.info(
            "  %-22s acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f",
            name,
            metrics.get("accuracy", float("nan")),
            metrics.get("precision", float("nan")),
            metrics.get("recall", float("nan")),
            metrics.get("f1", float("nan")),
        )
