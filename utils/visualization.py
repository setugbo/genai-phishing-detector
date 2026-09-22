"""
Shared visualization utilities for evaluation and explainability.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from config.config import CLASS_NAMES, EVAL_FIGURES_DIR

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save figure. If None, displays.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """
    Plot multi-class ROC curves with macro-average.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities (shape: n_samples x n_classes).
        class_names: List of class names.
        save_path: Path to save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(
            fpr[i], tpr[i],
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            linewidth=2,
        )

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, linestyle="--", color="navy", linewidth=2.5,
            label=f"Macro-average (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")

    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """
    Plot multi-class Precision-Recall curves.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        class_names: List of class names.
        save_path: Path to save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    precision, recall, avg_prec = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(
            recall[i], precision[i],
            label=f"{class_names[i]} (AP = {avg_prec[i]:.3f})",
            linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"PR curves saved to {save_path}")

    return fig


def plot_probability_distribution(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Prediction Probability Distribution",
) -> plt.Figure:
    """
    Plot prediction probability distributions per class.

    Args:
        y_prob: Predicted probabilities.
        y_true: Ground truth labels.
        class_names: List of class names.
        save_path: Path to save figure.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))

    for i in range(n_classes):
        mask = y_true == i
        ax = axes[i] if n_classes > 1 else axes
        ax.hist(y_prob[mask, i], bins=20, alpha=0.7, color=f"C{i}",
                edgecolor="white", label=f"True {class_names[i]}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"{class_names[i]}")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Probability distribution saved to {save_path}")

    return fig
