"""
Training Module for GenAI Phishing Detector.

Fine-tunes DistilBERT on the phishing dataset using HuggingFace Trainer API.
Computes Accuracy, Precision, Recall, F1, and AUPRC.
"""

import os
import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
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

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    MODEL_NAME,
    MAX_LENGTH,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    EVALUATION_STRATEGY,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
    MODEL_SAVE_DIR,
    TOKENIZER_SAVE_DIR,
    RAW_DATASET_CSV,
    CLASS_NAMES,
)
from preprocessing.preprocessing import TextPreprocessor

set_seed(RANDOM_SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute classification metrics from predictions.

    Args:
        eval_pred: Tuple of (predictions, labels).

    Returns:
        Dictionary with accuracy, precision, recall, f1, auprc.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Standard metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # AUPRC (one-vs-rest for multi-class)
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
# Dataset preparation
# ---------------------------------------------------------------------------

def load_and_split_dataset(
    csv_path: str = str(RAW_DATASET_CSV),
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
) -> DatasetDict:
    """
    Load CSV and split into train/val/test sets.

    Args:
        csv_path: Path to dataset CSV with 'text' and 'label' columns.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Val distribution:\n{val_df['label'].value_counts().sort_index()}")
    print(f"Test distribution:\n{test_df['label'].value_counts().sort_index()}")

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

    return dataset_dict


def tokenize_dataset(dataset: DatasetDict, tokenizer: DistilBertTokenizerFast) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    Args:
        dataset: DatasetDict with 'text' and 'label' columns.
        tokenizer: DistilBERT tokenizer.

    Returns:
        Tokenized DatasetDict.
    """
    def _tokenize_fn(batch: Dict[str, list]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(_tokenize_fn, batched=True)
    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in tokenized.column_names]
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    csv_path: str = str(RAW_DATASET_CSV),
    model_name: str = MODEL_NAME,
    output_dir: str = str(MODEL_SAVE_DIR),
    tokenizer_save_dir: str = str(TOKENIZER_SAVE_DIR),
    num_epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    resume_from_checkpoint: Optional[str] = None,
) -> Tuple[Trainer, DatasetDict, DistilBertTokenizerFast]:
    """
    Full training pipeline: load data, tokenize, train, save model.

    Args:
        csv_path: Path to dataset CSV.
        model_name: Pretrained model identifier.
        output_dir: Directory to save the trained model.
        tokenizer_save_dir: Directory to save the tokenizer.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: AdamW learning rate.
        resume_from_checkpoint: Optional checkpoint path to resume from.

    Returns:
        Tuple of (trainer, tokenized_datasets, tokenizer).
    """
    print("=" * 60)
    print("GenAI Phishing Detector — Training Pipeline")
    print("=" * 60)

    # 1. Load and split
    print("\n[1/5] Loading and splitting dataset ...")
    dataset = load_and_split_dataset(csv_path)

    # 2. Load tokenizer and model
    print("\n[2/5] Loading DistilBERT tokenizer and model ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label={i: label for i, label in enumerate(CLASS_NAMES)},
        label2id={label: i for i, label in enumerate(CLASS_NAMES)},
    )

    # 3. Tokenize
    print("\n[3/5] Tokenizing dataset ...")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    # 4. Training arguments
    print("\n[4/5] Configuring training arguments ...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
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

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set:")
    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save
    print(f"\nSaving model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"Saving tokenizer to {tokenizer_save_dir} ...")

    return trainer, tokenized_datasets, tokenizer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer, datasets, tokenizer = train_model(csv_path=str(RAW_DATASET_CSV))
