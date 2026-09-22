"""
Evaluation Module for GenAI Phishing Detector.

Generates:
    - Confusion matrix (raw + normalized)
    - ROC curves (per-class + macro-average)
    - Precision-Recall curves (per-class + macro-average)
    - Classification report (precision, recall, f1 per class)
    - Probability distribution analysis
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    MODEL_NAME,
    MODEL_SAVE_DIR,
    TOKENIZER_SAVE_DIR,
    RAW_DATASET_CSV,
    EVAL_REPORT_DIR,
    EVAL_FIGURES_DIR,
    CLASS_NAMES,
    MAX_LENGTH,
    BATCH_SIZE,
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_probability_distribution,
)


def load_model_and_tokenizer(
    model_path: str = str(MODEL_SAVE_DIR),
    tokenizer_path: str = str(TOKENIZER_SAVE_DIR),
) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """
    Load trained model and tokenizer from disk.

    Args:
        model_path: Path to saved model directory.
        tokenizer_path: Path to saved tokenizer directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading model from: {model_path}")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def load_test_data(
    csv_path: str = str(RAW_DATASET_CSV),
    test_ratio: float = 0.1,
) -> Tuple[List[str], np.ndarray]:
    """
    Load the test split from the dataset.

    Args:
        csv_path: Path to dataset CSV.
        test_ratio: Fraction of data to use as test.

    Returns:
        Tuple of (texts, labels).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    test_start = int(n * (1 - test_ratio))
    test_df = df.iloc[test_start:].reset_index(drop=True)

    print(f"Test samples: {len(test_df)}")
    print(f"Test distribution:\n{test_df['label'].value_counts().sort_index()}")

    return test_df["text"].tolist(), test_df["label"].values


def predict(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return predictions + probabilities.

    Args:
        model: Fine-tuned DistilBERT model.
        tokenizer: DistilBERT tokenizer.
        texts: List of input text strings.
        batch_size: Batch size for inference.

    Returns:
        Tuple of (predicted_class_indices, probabilities).
    """
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    predictions = np.argmax(probs, axis=1)

    return predictions, probs


def evaluate_model(
    model_path: str = str(MODEL_SAVE_DIR),
    tokenizer_path: str = str(TOKENIZER_SAVE_DIR),
    csv_path: str = str(RAW_DATASET_CSV),
    output_dir: str = str(EVAL_REPORT_DIR),
    figures_dir: str = str(EVAL_FIGURES_DIR),
) -> Dict[str, float]:
    """
    End-to-end evaluation: load model, run inference, generate all reports.

    Args:
        model_path: Path to saved model.
        tokenizer_path: Path to saved tokenizer.
        csv_path: Path to dataset CSV.
        output_dir: Directory for evaluation reports.
        figures_dir: Directory for evaluation figures.

    Returns:
        Dictionary of overall metrics.
    """
    output_path = Path(output_dir)
    figures_path = Path(figures_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GenAI Phishing Detector — Evaluation Pipeline")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading model and test data ...")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    texts, y_true = load_test_data(csv_path)

    # 2. Run inference
    print("\n[2/4] Running inference ...")
    y_pred, y_prob = predict(model, tokenizer, texts)

    # 3. Generate metrics
    print("\n[3/4] Generating metrics and plots ...")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_path = output_path / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nClassification report saved to: {report_path}")

    # Print report
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Confusion matrix (image)
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=figures_path / "confusion_matrix.png",
    )

    # Confusion matrix (text) -- so zero-misclassification can be verified
    # directly as text, not only as an embedded image.
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = CLASS_NAMES[:n_classes]
    label_w = max(len(l) for l in labels)
    cell_w = max(len(str(cm.max())), 6) + 2
    col_hdrs = [f"P:{l}" for l in labels]

    sep = "+" + "-" * (label_w + 2) + "+" + "+".join(
        "-" * cell_w for _ in labels
    ) + "+"
    cm_lines = [sep]
    cm_lines.append(
        "|" + " " * (label_w + 2) + "|"
        + "|".join(f"{h:^{cell_w}}" for h in col_hdrs) + "|"
    )
    cm_lines.append(sep.replace("-", "="))
    for i in range(n_classes):
        row_lbl = f"T:{labels[i]}"
        cm_lines.append(
            f"|{row_lbl:>{label_w + 2}}|"
            + "|".join(f"{cm[i][j]:^{cell_w}}" for j in range(n_classes))
            + "|"
        )
    cm_lines.append(sep)

    total_correct = int(np.trace(cm))
    total_samples = int(cm.sum())
    overall_acc = total_correct / total_samples if total_samples else 0
    off_diagonal = total_samples - total_correct

    cm_lines.append("")
    cm_lines.append("Per-class accuracy:")
    for i in range(n_classes):
        total = int(cm[i].sum())
        correct = int(cm[i][i])
        acc = correct / total if total else 0
        cm_lines.append(f"  {labels[i]:{label_w}}: {correct}/{total} = {acc:.4f}")
    cm_lines.append(
        f"\n  Overall accuracy: {total_correct}/{total_samples} = {overall_acc:.4f}"
    )
    if off_diagonal == 0:
        cm_lines.append(f"\n  CONFIRMED: Zero misclassifications across all classes.")
    else:
        cm_lines.append(
            f"\n  {off_diagonal} misclassification(s) "
            f"({off_diagonal}/{total_samples})."
        )

    cm_text = "\n".join(cm_lines)
    print("\nConfusion Matrix (text):\n")
    print(cm_text)

    cm_text_path = output_path / "confusion_matrix.txt"
    with open(cm_text_path, "w", encoding="utf-8") as f:
        f.write(cm_text)
    print(f"\nConfusion matrix (text) saved to: {cm_text_path}")

    # ROC curves
    plot_roc_curves(
        y_true, y_prob,
        save_path=figures_path / "roc_curves.png",
    )

    # PR curves
    plot_precision_recall_curves(
        y_true, y_prob,
        save_path=figures_path / "pr_curves.png",
    )

    # Probability distribution
    plot_probability_distribution(
        y_prob, y_true,
        save_path=figures_path / "probability_distribution.png",
    )

    # 4. Summary metrics
    print("\n[4/4] Summary metrics:")
    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    metrics_path = output_path / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics summary saved to: {metrics_path}")

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
    per_class_path = output_path / "per_class_metrics.json"
    with open(per_class_path, "w") as f:
        json.dump(per_class, f, indent=2)

    print("\nEvaluation complete. All reports and figures generated.")
    return metrics


if __name__ == "__main__":
    evaluate_model()
