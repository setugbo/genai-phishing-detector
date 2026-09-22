"""
Evaluation Script for Template-Aware Split Experiment.

Generates:
  - Accuracy, Precision, Recall, F1, AUPRC
  - Confusion matrix (image + text)
  - Classification report
  - Training history curves
  - Near-duplicate analysis
  - Comparison with original experiment

Usage:
    python experiments/template_aware_split/evaluate.py
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_NAME,
    MAX_LENGTH,
    CLASS_NAMES,
    RANDOM_SEED,
)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "template_aware_split"
DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models" / "phishing_model"
TOKENIZER_DIR = EXPERIMENT_DIR / "models" / "tokenizer"
REPORT_DIR = EXPERIMENT_DIR / "reports"
FIG_DIR = EXPERIMENT_DIR / "figures"


# ---------------------------------------------------------------------------
# Load model and run inference
# ---------------------------------------------------------------------------

def load_model_and_data():
    """Load trained model and test data."""
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    test_df["label"] = test_df["label"].astype(int)

    return model, tokenizer, test_df, device


def run_inference(model, tokenizer, texts, device, batch_size=16):
    """Run batched inference and return predictions + probabilities."""
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        all_logits.append(out.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ---------------------------------------------------------------------------
# Confusion matrix (text)
# ---------------------------------------------------------------------------

def confusion_matrix_text(y_true, y_pred) -> str:
    """Generate bordered ASCII confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    labels = CLASS_NAMES[:n]
    lw = max(len(l) for l in labels)
    cw = max(len(str(cm.max())), 6) + 2
    col_hdrs = [f"P:{l}" for l in labels]
    row_hdrs = [f"T:{l}" for l in labels]

    sep = "+" + "-" * (lw + 2) + "+" + "+".join("-" * cw for _ in labels) + "+"
    lines = [sep]
    lines.append("|" + " " * (lw + 2) + "|" +
                 "|".join(f"{h:^{cw}}" for h in col_hdrs) + "|")
    lines.append(sep.replace("-", "="))
    for i in range(n):
        lines.append(
            f"|{row_hdrs[i]:>{lw + 2}}|" +
            "|".join(f"{cm[i][j]:^{cw}}" for j in range(n)) + "|"
        )
    lines.append(sep)

    total = int(cm.sum())
    correct = int(np.trace(cm))
    off_diag = total - correct
    lines.append(f"\nTotal test samples: {total}")
    lines.append(f"Correctly classified: {correct}")
    lines.append(f"Misclassified: {off_diag}")
    lines.append(f"Overall accuracy: {correct}/{total} = {correct/total:.4f}")

    if off_diag == 0:
        lines.append("CONFIRMED: Zero misclassifications across all classes.")
    else:
        lines.append(f"{off_diag} misclassification(s) found.")

    # Per-class
    lines.append("\nPer-class accuracy:")
    for i in range(n):
        t = int(cm[i].sum())
        c = int(cm[i][i])
        lines.append(f"  {labels[i]:{lw}}: {c}/{t} = {c/t:.4f}" if t else
                      f"  {labels[i]:{lw}}: 0/0")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Near-duplicate analysis
# ---------------------------------------------------------------------------

def near_duplicate_analysis(train_df, test_df, val_df=None):
    """Compute n-gram Jaccard similarity between splits."""
    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE ANALYSIS (Template-Aware Split)")
    print("=" * 60)

    def _ngrams(text, n=3):
        t = re.sub(r"\s+", " ", text.lower()).strip()
        return set(t[i:i+n] for i in range(max(len(t) - n + 1, 1)))

    def _jaccard(a, b):
        return len(a & b) / len(a | b) if (a | b) else 1.0

    comparisons = [
        ("Train vs Test", train_df, test_df),
        ("Train vs Validation", train_df, val_df),
        ("Validation vs Test", val_df, test_df),
    ]

    results = {}
    for label, df_a, df_b in comparisons:
        if df_b is None:
            continue
        texts_a = df_a["text"].tolist()
        texts_b = df_b["text"].tolist()
        ngrams_a = [_ngrams(t) for t in texts_a]
        ngrams_b = [_ngrams(t) for t in texts_b]

        max_sim = 0.0
        max_pair = (0, 0)
        above_80 = 0
        total_pairs = len(texts_a) * len(texts_b)

        for i, a in enumerate(ngrams_a):
            for j, b in enumerate(ngrams_b):
                s = _jaccard(a, b)
                if s > max_sim:
                    max_sim = s
                    max_pair = (i, j)
                if s >= 0.80:
                    above_80 += 1

        # Check shared template families
        families_a = set(df_a["template_family_id"].unique())
        families_b = set(df_b["template_family_id"].unique())
        shared = families_a & families_b

        results[label] = {
            "max_similarity": round(max_sim, 6),
            "pairs_at_or_above_0.80": above_80,
            "total_pairs": total_pairs,
            "shared_template_families": len(shared),
            "shared_family_ids": sorted(shared),
        }

        print(f"\n  {label}:")
        print(f"    Max Jaccard similarity:  {max_sim:.4f}")
        print(f"    Pairs >= 0.80:           {above_80}")
        print(f"    Shared template families:{len(shared)}")

    return results


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(trainer):
    """Extract training history from trainer and plot curves."""
    history = trainer.state.log_history
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_loss = [h["loss"] for h in history if "loss" in h and "eval_loss" not in h]
    eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h]
    eval_acc = [h.get("eval_accuracy", h.get("eval_f1")) for h in history
                if "eval_loss" in h]

    epochs_t = list(range(1, len(train_loss) + 1))
    epochs_v = list(range(1, len(eval_loss) + 1))

    # Training loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_t, train_loss, "b-o", linewidth=2, markersize=8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training Loss Curve (Template-Aware Split)",
              fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.xticks(epochs_t)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Validation accuracy / F1
    eval_f1 = [h.get("eval_f1") for h in history if "eval_loss" in h]
    if eval_acc[0] and eval_f1[0]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_v, eval_acc, "g-s", linewidth=2, markersize=8,
                 label="Validation Accuracy")
        plt.plot(epochs_v, eval_f1, "m-d", linewidth=2, markersize=8,
                 label="Validation F1")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Validation Accuracy & F1 (Template-Aware Split)",
                  fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.xticks(epochs_v)
        plt.ylim(0.0, 1.05)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "validation_accuracy_curve.png", dpi=200,
                    bbox_inches="tight")
        plt.close()

    print(f"  Training curves saved to: {FIG_DIR}")

    # Print epoch-by-epoch values
    print("\n  Epoch-by-epoch training history:")
    for h in history:
        if "loss" in h and "eval_loss" not in h:
            epoch = h.get("epoch", "?")
            print(f"    Epoch {epoch}: train_loss={h['loss']:.4f}")
        if "eval_loss" in h:
            epoch = h.get("epoch", "?")
            parts = [f"eval_loss={h['eval_loss']:.4f}"]
            if "eval_accuracy" in h:
                parts.append(f"eval_acc={h['eval_accuracy']:.4f}")
            if "eval_f1" in h:
                parts.append(f"eval_f1={h['eval_f1']:.4f}")
            if "eval_auprc" in h:
                parts.append(f"eval_auprc={h['eval_auprc']:.4f}")
            print(f"    Epoch {epoch}: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Comparison with original
# ---------------------------------------------------------------------------

def comparison_report(new_metrics, new_overlap, original_metrics=None):
    """Print and save comparison table."""
    print("\n" + "=" * 70)
    print("TEMPLATE-AWARE EVALUATION SUMMARY")
    print("=" * 70)

    if original_metrics:
        print("\nOriginal random split:")
        print(f"  Accuracy  = {original_metrics.get('accuracy', '?')}")
        print(f"  Precision = {original_metrics.get('precision', '?')}")
        print(f"  Recall    = {original_metrics.get('recall', '?')}")
        print(f"  F1        = {original_metrics.get('f1', '?')}")
        print(f"  AUPRC     = {original_metrics.get('auprc', '?')}")
        if "overlap" in original_metrics:
            ov = original_metrics["overlap"]
            print(f"\n  Original overlap findings:")
            print(f"    Max similarity = {ov.get('max_similarity', '?')}")
            print(f"    Shared families = {ov.get('shared_families', '?')}")
            print(f"    Potential contamination = "
                  f"{ov.get('contamination_rate', '?')}")

    print(f"\n{'-' * 70}")
    print("NEW template-aware split:")

    # Load split sizes
    for split_name in ["train", "validation", "test"]:
        csv_path = DATA_DIR / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {split_name}: {len(df)} samples")

    # Shared families
    print(f"\n  Shared train/test template families = "
          f"{new_overlap.get('train vs test', {}).get('shared_template_families', '?')}")
    print(f"  Shared train/val template families = "
          f"{new_overlap.get('train vs validation', {}).get('shared_template_families', '?')}")
    print(f"  Shared val/test template families = "
          f"{new_overlap.get('validation vs test', {}).get('shared_template_families', '?')}")

    print(f"\nNew results:")
    print(f"  Accuracy  = {new_metrics.get('accuracy', '?')}")
    print(f"  Precision = {new_metrics.get('precision', '?')}")
    print(f"  Recall    = {new_metrics.get('recall', '?')}")
    print(f"  F1        = {new_metrics.get('f1', '?')}")
    print(f"  AUPRC     = {new_metrics.get('auprc', '?')}")

    tt = new_overlap.get("Train vs Test", {})
    print(f"\n  New overlap findings:")
    print(f"    Max similarity = {tt.get('max_similarity', '?')}")
    print(f"    Shared families = {tt.get('shared_template_families', '?')}")
    print(f"    Pairs >= 0.80 = {tt.get('pairs_at_or_above_0.80', '?')}")

    # Comparison table
    if original_metrics:
        print(f"\n{'=' * 70}")
        print("COMPARISON: Original vs Template-Aware")
        print(f"{'=' * 70}")
        header = f"{'Metric':<20} {'Original':>15} {'Template-Aware':>15}"
        print(header)
        print("-" * 55)
        for key in ["accuracy", "precision", "recall", "f1", "auprc"]:
            orig = original_metrics.get(key, "?")
            new = new_metrics.get(key, "?")
            if isinstance(orig, float):
                orig = f"{orig:.4f}"
            if isinstance(new, float):
                new = f"{new:.4f}"
            print(f"{key.upper():<20} {orig:>15} {new:>15}")
        print("=" * 70)

    # Save to JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "template_aware_metrics": new_metrics,
        "near_duplicate_analysis": new_overlap,
    }
    if original_metrics:
        report["original_metrics"] = original_metrics
    path = REPORT_DIR / "comparison_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Comparison report saved to: {path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(trainer=None, original_metrics=None):
    """Full evaluation pipeline."""
    print("=" * 60)
    print("TEMPLATE-AWARE SPLIT — Evaluation Pipeline")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model and data
    print("\n[1/5] Loading model and test data ...")
    model, tokenizer, test_df, device = load_model_and_data()

    # 2. Inference
    print("\n[2/5] Running inference ...")
    texts = test_df["text"].tolist()
    y_true = test_df["label"].values
    y_pred, y_prob = run_inference(model, tokenizer, texts, device)

    # 3. Metrics
    print("\n[3/5] Computing metrics ...")
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # AUPRC (one-vs-rest weighted)
    n_classes = 3
    labels_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        auprc = average_precision_score(labels_bin, y_prob, average="weighted")
    except Exception:
        auprc = 0.0

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auprc": round(float(auprc), 4),
    }

    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")

    # Classification report
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    print(f"\n{report_text}")
    with open(REPORT_DIR / "classification_report.txt", "w") as f:
        f.write(report_text)

    # Confusion matrix
    print("\n[4/5] Generating confusion matrix ...")
    cm_text = confusion_matrix_text(y_true, y_pred)
    print(f"\n{cm_text}")
    with open(REPORT_DIR / "confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write(cm_text)

    # Confusion matrix image
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    plt.title("Confusion Matrix — Template-Aware Split", fontsize=14,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 5. Near-duplicate analysis
    print("\n[5/5] Near-duplicate analysis ...")
    train_df = pd.read_csv(DATA_DIR / "train_metadata.csv")
    val_df = pd.read_csv(DATA_DIR / "validation_metadata.csv")
    test_meta_df = pd.read_csv(DATA_DIR / "test_metadata.csv")
    overlap = near_duplicate_analysis(train_df, test_meta_df, val_df)

    # Training curves
    if trainer is not None:
        print("\nGenerating training curves ...")
        plot_training_curves(trainer)

    # Save metrics
    with open(REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Comparison
    comparison_report(metrics, overlap, original_metrics)

    return metrics, overlap


if __name__ == "__main__":
    run_evaluation()
