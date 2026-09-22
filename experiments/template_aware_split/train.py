"""
Training Script for Template-Aware Split Experiment.

Retrains DistilBERT on the group-based split with zero template overlap.
Same architecture and hyperparameters as the original experiment.

Usage:
    python experiments/template_aware_split/train.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_NAME,
    MAX_LENGTH,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LOGGING_STEPS,
    CLASS_NAMES,
)

set_seed(RANDOM_SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "template_aware_split"
DATA_DIR = EXPERIMENT_DIR / "data"
MODEL_DIR = EXPERIMENT_DIR / "models" / "phishing_model"
TOKENIZER_DIR = EXPERIMENT_DIR / "models" / "tokenizer"
REPORT_DIR = EXPERIMENT_DIR / "reports"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, AUPRC from predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    n_classes = logits.shape[1]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        auprc = average_precision_score(labels_bin, probs, average="weighted")
    except Exception:
        auprc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auprc": round(auprc, 4),
    }


# ---------------------------------------------------------------------------
# Dataset loading from split CSVs
# ---------------------------------------------------------------------------

def load_split_csvs(data_dir: Path) -> DatasetDict:
    """Load train/val/test CSVs produced by split_and_validate.py."""
    splits = {}
    for name in ["train", "validation", "test"]:
        csv_path = data_dir / f"{name}.csv"
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        splits[name] = Dataset.from_pandas(df[["text", "label"]])
        print(f"  {name}: {len(df)} samples")

    return DatasetDict(splits)


def tokenize_dataset(dataset: DatasetDict, tokenizer: DistilBertTokenizerFast) -> DatasetDict:
    """Tokenize all splits."""
    def _tokenize_fn(batch: Dict[str, list]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(_tokenize_fn, batched=True)
    cols_to_remove = [c for c in ["text", "__index_level_0__"]
                      if c in tokenized.column_names]
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_template_aware() -> Tuple[Trainer, DatasetDict, DistilBertTokenizerFast]:
    """Full training pipeline on template-aware split."""
    print("=" * 60)
    print("TEMPLATE-AWARE SPLIT — Training Pipeline")
    print("=" * 60)

    # 1. Load splits
    print("\n[1/5] Loading template-aware splits ...")
    dataset = load_split_csvs(DATA_DIR)

    # 2. Load model and tokenizer
    print("\n[2/5] Loading DistilBERT tokenizer and model ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={i: label for i, label in enumerate(CLASS_NAMES)},
        label2id={label: i for i, label in enumerate(CLASS_NAMES)},
    )

    # 3. Tokenize
    print("\n[3/5] Tokenizing dataset ...")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    # 4. Training arguments (same hyperparameters as original)
    print("\n[4/5] Configuring training arguments ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=RANDOM_SEED,
        remove_unused_columns=False,
    )

    # 5. Trainer
    print("\n[5/5] Initialising Trainer and starting training ...\n")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final evaluation on template-aware test set:")
    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save model and tokenizer
    print(f"\nSaving model to {MODEL_DIR} ...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print(f"Saving tokenizer to {TOKENIZER_DIR} ...")

    # Save test metrics
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"Test metrics saved to: {metrics_path}")

    return trainer, tokenized_datasets, tokenizer


if __name__ == "__main__":
    trainer, datasets, tokenizer = train_template_aware()
