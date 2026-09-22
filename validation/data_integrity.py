"""
Data Integrity Validation for GenAI Phishing Detector.

Implements three remediation checks recommended during code review:

  (i)   Near-duplicate check between train and test AI-generated phishing
        samples using n-gram overlap (Jaccard) and optional embedding cosine
        similarity.
  (ii)  Group-based split validation: extracts template skeletons from
        AI-generated samples and checks whether template families span
        both train and test sets.
  (iii) Confusion matrix text reproduction so zero-misclassification
        results can be verified directly (not only as an embedded image).

Usage:
    python validation/data_integrity.py [--dataset PATH] [--fast] [--json PATH]
"""

import argparse
import json
import os
import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    RAW_DATASET_CSV,
    RANDOM_SEED,
    CLASS_NAMES,
    VALIDATION_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLOT_PATTERN = re.compile(
    r"NGN[\d,]+|"                # currency amounts
    r"\b\d{9,15}\b|"             # account numbers
    r"\b\d{2}-[A-Z]{3}-\d{4}\b|" # dates like 15-Jan-2025
    r"\b\d{4}-\d{2}-\d{2}\b|"   # dates like 2025-01-15
    r"https?://\S+|"             # URLs
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|"  # emails
    r"\b0\d{10}\b|"              # Nigerian phone numbers
    r"\*\d{3,4}\#"              # USSD codes
    r"\b[A-Z]{2}\d{8,}\b"       # reference numbers
)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _to_skeleton(text: str) -> str:
    """Strip slot values to expose the template skeleton."""
    text = _normalise(text)
    text = _SLOT_PATTERN.sub(" <SLOT> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _skeleton_hash(skeleton: str) -> str:
    return hashlib.sha256(skeleton.encode()).hexdigest()[:16]


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Character-level n-grams."""
    text = _normalise(text)
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# (i) Near-duplicate check
# ---------------------------------------------------------------------------

def check_near_duplicates(
    train_texts: List[str],
    test_texts: List[str],
    threshold: float = 0.8,
) -> Dict:
    """
    Check for near-duplicate samples across train/test splits.

    Uses character 3-gram Jaccard similarity as the fast method.
    Optionally computes DistilBERT [CLS] embedding cosine similarity.

    Args:
        train_texts: Training set texts (AI-generated phishing only).
        test_texts: Test set texts (AI-generated phishing only).
        threshold: Jaccard threshold to flag a pair as near-duplicate.

    Returns:
        Dictionary with n-gram and embedding results.
    """
    print("\n" + "=" * 60)
    print("(i) NEAR-DUPLICATE CHECK")
    print("=" * 60)

    # --- N-gram overlap (Jaccard) ---
    print("\n  Computing character 3-gram Jaccard similarity ...")
    train_ngrams = [_char_ngrams(t) for t in train_texts]
    test_ngrams = [_char_ngrams(t) for t in test_texts]

    max_jaccard = 0.0
    max_pair = (-1, -1)
    above_threshold = 0
    total_pairs = len(train_texts) * len(test_texts)
    all_scores: List[float] = []

    for i, tr_ng in enumerate(train_ngrams):
        for j, te_ng in enumerate(test_ngrams):
            score = _jaccard(tr_ng, te_ng)
            all_scores.append(score)
            if score > max_jaccard:
                max_jaccard = score
                max_pair = (i, j)
            if score >= threshold:
                above_threshold += 1

    scores_arr = np.array(all_scores)

    ngram_result = {
        "method": "char_3gram_jaccard",
        "total_pairs_compared": total_pairs,
        "max_similarity": round(float(max_jaccard), 6),
        "mean_similarity": round(float(scores_arr.mean()), 6),
        "median_similarity": round(float(np.median(scores_arr)), 6),
        "pairs_at_or_above_threshold": above_threshold,
        "threshold": threshold,
        "max_pair": {
            "train_index": int(max_pair[0]),
            "test_index": int(max_pair[1]),
            "train_text_preview": _normalise(train_texts[max_pair[0]])[:120],
            "test_text_preview": _normalise(test_texts[max_pair[1]])[:120],
        },
    }

    print(f"\n  Pairs compared:           {total_pairs}")
    print(f"  Max Jaccard similarity:   {max_jaccard:.4f}")
    print(f"  Mean Jaccard similarity:  {scores_arr.mean():.4f}")
    print(f"  Median Jaccard similarity:{np.median(scores_arr):.4f}")
    print(f"  Pairs >= {threshold:.2f}:        {above_threshold}")
    if above_threshold > 0:
        print(f"\n  !! WARNING: {above_threshold} train/test pair(s) at or "
              f"above the {threshold:.2f} threshold.")
        print(f"     Most similar pair (train[{max_pair[0]}], test[{max_pair[1]}]):")
        print(f"       Train: {ngram_result['max_pair']['train_text_preview']}...")
        print(f"       Test:  {ngram_result['max_pair']['test_text_preview']}...")
    else:
        print(f"\n  OK: No n-gram pairs above the {threshold:.2f} threshold.")

    # --- Embedding cosine similarity (optional) ---
    embedding_result = _check_embedding_duplicates(train_texts, test_texts)

    return {
        "ngram_jaccard": ngram_result,
        "embedding_cosine": embedding_result,
    }


def _check_embedding_duplicates(
    train_texts: List[str],
    test_texts: List[str],
    sample_size: int = 200,
) -> Optional[Dict]:
    """Compute [CLS] embedding cosine similarity (optional, slower)."""
    try:
        import torch
        from transformers import DistilBertModel, DistilBertTokenizerFast
        from config.config import MODEL_NAME, MAX_LENGTH
    except ImportError:
        print("\n  Skipping embedding check (torch/transformers not available).")
        return None

    print("\n  Computing DistilBERT [CLS] embedding cosine similarity ...")
    print(f"  (sampling up to {sample_size} pairs per side for speed)")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _embed(texts: List[str]) -> np.ndarray:
        embeddings = []
        bs = 32
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            enc = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
        return np.vstack(embeddings)

    # Subsample if large
    tr_sample = train_texts[:sample_size]
    te_sample = test_texts[:sample_size]

    tr_emb = _embed(tr_sample)
    te_emb = _embed(te_sample)

    # L2-normalise
    tr_emb = tr_emb / np.linalg.norm(tr_emb, axis=1, keepdims=True)
    te_emb = te_emb / np.linalg.norm(te_emb, axis=1, keepdims=True)

    sim_matrix = tr_emb @ te_emb.T
    max_cos = float(sim_matrix.max())
    max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    mean_cos = float(sim_matrix.mean())
    above_90 = int((sim_matrix >= 0.90).sum())
    above_95 = int((sim_matrix >= 0.95).sum())

    result = {
        "method": "distilbert_cls_cosine",
        "train_sampled": len(tr_sample),
        "test_sampled": len(te_sample),
        "max_cosine_similarity": round(max_cos, 6),
        "mean_cosine_similarity": round(mean_cos, 6),
        "pairs_at_or_above_0.90": above_90,
        "pairs_at_or_above_0.95": above_95,
        "max_pair": {
            "train_index": int(max_idx[0]),
            "test_index": int(max_idx[1]),
            "train_text_preview": _normalise(tr_sample[max_idx[0]])[:120],
            "test_text_preview": _normalise(te_sample[max_idx[1]])[:120],
        },
    }

    print(f"  Max cosine similarity:    {max_cos:.4f}")
    print(f"  Mean cosine similarity:   {mean_cos:.4f}")
    print(f"  Pairs >= 0.90:            {above_90}")
    print(f"  Pairs >= 0.95:            {above_95}")

    if above_95 > 0:
        print(f"\n  !! WARNING: {above_95} pair(s) above 0.95 cosine similarity.")
        print(f"     Most similar: train[{max_idx[0]}], test[{max_idx[1]}]")
    elif above_90 > 0:
        print(f"\n  NOTE: {above_90} pair(s) above 0.90 cosine similarity.")
    else:
        print(f"\n  OK: No embedding pairs above 0.90 cosine similarity.")

    return result


# ---------------------------------------------------------------------------
# (ii) Group-based split check
# ---------------------------------------------------------------------------

def check_group_split(
    train_texts: List[str],
    test_texts: List[str],
) -> Dict:
    """
    Check whether template families span both train and test splits.

    Extracts a template skeleton by removing slot values (names, amounts,
    dates, URLs, phone numbers, account numbers) and hashing the remainder.
    Samples from the same template produce the same skeleton hash.

    Args:
        train_texts: Training set texts (AI-generated phishing only).
        test_texts: Test set texts (AI-generated phishing only).

    Returns:
        Dictionary with group-based split analysis.
    """
    print("\n" + "=" * 60)
    print("(ii) GROUP-BASED SPLIT CHECK (Template Families)")
    print("=" * 60)

    train_skeletons = [_to_skeleton(t) for t in train_texts]
    test_skeletons = [_to_skeleton(t) for t in test_texts]

    train_hashes = [_skeleton_hash(s) for s in train_skeletons]
    test_hashes = [_skeleton_hash(s) for s in test_skeletons]

    # Group by skeleton hash
    train_groups: Dict[str, List[int]] = defaultdict(list)
    for i, h in enumerate(train_hashes):
        train_groups[h].append(i)

    test_groups: Dict[str, List[int]] = defaultdict(list)
    for i, h in enumerate(test_hashes):
        test_groups[h].append(i)

    all_hashes = set(train_groups.keys()) | set(test_groups.keys())
    total_families = len(all_hashes)

    # Families that appear in BOTH train and test
    shared_hashes = set(train_groups.keys()) & set(test_groups.keys())
    split_families = len(shared_hashes)

    # Families only in train or only in test
    train_only = set(train_groups.keys()) - set(test_groups.keys())
    test_only = set(test_groups.keys()) - set(train_groups.keys())

    # How many samples are involved?
    train_in_split = sum(len(train_groups[h]) for h in shared_hashes)
    test_in_split = sum(len(test_groups[h]) for h in shared_hashes)

    # Find the specific skeletons that appear on both sides
    split_examples = []
    for h in sorted(shared_hashes):
        split_examples.append({
            "skeleton_hash": h,
            "train_count": len(train_groups[h]),
            "test_count": len(test_groups[h]),
            "skeleton_preview": _to_skeleton(
                train_texts[train_groups[h][0]]
            )[:100],
        })

    result = {
        "total_unique_skeletons": total_families,
        "skeletons_in_both_splits": split_families,
        "skeletons_train_only": len(train_only),
        "skeletons_test_only": len(test_only),
        "train_samples_in_shared_families": train_in_split,
        "test_samples_in_shared_families": test_in_split,
        "total_train_samples": len(train_texts),
        "total_test_samples": len(test_texts),
        "split_contamination_rate": round(
            test_in_split / len(test_texts) if test_texts else 0, 4
        ),
        "examples": split_examples[:20],
    }

    print(f"\n  Total unique template skeletons: {total_families}")
    print(f"  Skeletons in BOTH train & test:   {split_families}")
    print(f"  Skeletons only in train:          {len(train_only)}")
    print(f"  Skeletons only in test:           {len(test_only)}")
    print(f"  Train samples from shared families: {train_in_split}/{len(train_texts)}")
    print(f"  Test samples from shared families:  {test_in_split}/{len(test_texts)}")
    contamination = test_in_split / len(test_texts) if test_texts else 0
    print(f"  Test-set contamination rate:       {contamination:.2%}")

    if split_families > 0:
        print(f"\n  !! FINDING: {split_families} template families appear in "
              f"both train and test splits.")
        print(f"     This means {test_in_split} of {len(test_texts)} test "
              f"samples share a template skeleton with training samples.")
        print(f"\n  Top shared skeletons:")
        for ex in split_examples[:10]:
            print(f"    Hash {ex['skeleton_hash']}: "
                  f"train={ex['train_count']}, test={ex['test_count']} "
                  f"  \"{ex['skeleton_preview']}...\"")
    else:
        print(f"\n  OK: No template families span both train and test.")

    return result


# ---------------------------------------------------------------------------
# (iii) Confusion matrix text reproduction
# ---------------------------------------------------------------------------

def print_confusion_matrix_text(y_true, y_pred) -> str:
    """
    Print and return the confusion matrix as a bordered ASCII text table.

    Args:
        y_true: Ground-truth label array.
        y_pred: Predicted label array.

    Returns:
        Formatted text string of the confusion matrix.
    """
    print("\n" + "=" * 60)
    print("(iii) CONFUSION MATRIX (Text Reproduction)")
    print("=" * 60)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = CLASS_NAMES[:n_classes]

    # Determine column widths
    label_width = max(len(l) for l in labels)
    label_width = max(label_width, 8)
    cell_width = max(len(f"{cm.max()}"), 6) + 2

    header = " " * (label_width + 2) + "".join(
        f"{'Predicted':^{cell_width}}" if i == 0 else f"{'':^{cell_width}}"
        for i in range(1)
    )
    # Actually, build column headers per class
    col_headers = [f"P:{l}" for l in labels]

    lines = []
    border = "+" + "-" * (label_width + 2) + "+" + "+".join(
        "-" * cell_width for _ in labels
    ) + "+"

    lines.append(border)
    header_row = "|" + " " * (label_width + 2) + "|"
    header_row += "|".join(f"{h:^{cell_width}}" for h in col_headers) + "|"
    lines.append(header_row)
    lines.append(border.replace("-", "="))

    for i in range(n_classes):
        row_label = f"T:{labels[i]}"
        row = f"|{row_label:>{label_width + 2}}|"
        row += "|".join(
            f"{cm[i][j]:^{cell_width}}"
            for j in range(n_classes)
        ) + "|"
        lines.append(row)

    lines.append(border)

    # Per-class accuracy
    lines.append("")
    lines.append("Per-class accuracy:")
    for i in range(n_classes):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        lines.append(f"  {labels[i]:{label_width}}: {correct}/{total} = {acc:.4f}")

    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    lines.append(f"\n  Overall accuracy: {total_correct}/{total_samples} = {overall_acc:.4f}")

    # Check for zero misclassification
    off_diagonal = int(cm.sum() - np.trace(cm))
    if off_diagonal == 0:
        lines.append(f"\n  RESULT: ZERO misclassifications ({off_diagonal} off-diagonal).")
    else:
        lines.append(f"\n  RESULT: {off_diagonal} misclassification(s) "
                      f"({off_diagonal}/{total_samples} error rate).")

    table_text = "\n".join(lines)
    print("\n" + table_text)

    return table_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_split(
    csv_path: str,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the exact same 80/10/10 split as training/train.py."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def run_validation(
    csv_path: str = str(RAW_DATASET_CSV),
    fast: bool = False,
    json_path: Optional[str] = None,
) -> Dict:
    """
    Run all three data integrity checks.

    Args:
        csv_path: Path to dataset CSV.
        fast: If True, skip the embedding cosine check (slower).
        json_path: Path to save the JSON report.

    Returns:
        Dictionary with all check results.
    """
    print("=" * 60)
    print("GENAI PHISHING DETECTOR — DATA INTEGRITY VALIDATION")
    print("=" * 60)
    print(f"  Dataset: {csv_path}")
    print(f"  Mode:    {'FAST (n-gram only)' if fast else 'FULL (n-gram + embeddings)'}")

    train_df, val_df, test_df = load_split(csv_path)

    print(f"\n  Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Random seed: {RANDOM_SEED}")

    # Filter to AI-generated phishing only (class 2)
    ai_train = train_df[train_df["label"] == 2]["text"].tolist()
    ai_test = test_df[test_df["label"] == 2]["text"].tolist()
    print(f"  AI-phishing samples: Train={len(ai_train)}, Test={len(ai_test)}")

    results = {}

    # (i) Near-duplicate check
    results["near_duplicate_check"] = check_near_duplicates(
        ai_train, ai_test, threshold=0.8
    )

    # (ii) Group-based split check
    results["group_split_check"] = check_group_split(ai_train, ai_test)

    # (iii) Confusion matrix text (requires model predictions)
    results["confusion_matrix_note"] = (
        "Confusion matrix text reproduction is integrated into "
        "evaluation/evaluate.py and the generate_all_evidence notebook. "
        "Run 'python evaluation/evaluate.py' after training to generate "
        "the text confusion matrix alongside the PNG plot."
    )

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    ng = results["near_duplicate_check"]["ngram_jaccard"]
    gs = results["group_split_check"]

    issues = 0
    if ng["pairs_at_or_above_threshold"] > 0:
        issues += 1
        print(f"  [!] Near-duplicates found: {ng['pairs_at_or_above_threshold']} "
              f"n-gram pair(s) >= {ng['threshold']}")
    else:
        print(f"  [OK] No near-duplicates (n-gram Jaccard)")

    if results["near_duplicate_check"]["embedding_cosine"]:
        ec = results["near_duplicate_check"]["embedding_cosine"]
        if ec["pairs_at_or_above_0.95"] > 0:
            issues += 1
            print(f"  [!] Near-duplicates found: {ec['pairs_at_or_above_0.95']} "
                  f"embedding pair(s) >= 0.95 cosine")
        elif ec["pairs_at_or_above_0.90"] > 0:
            print(f"  [NOTE] {ec['pairs_at_or_above_0.90']} embedding pair(s) "
                  f">= 0.90 cosine")
        else:
            print(f"  [OK] No near-duplicates (embedding cosine)")

    if gs["skeletons_in_both_splits"] > 0:
        issues += 1
        print(f"  [!] Template contamination: {gs['skeletons_in_both_splits']} "
              f"families span both splits "
              f"({gs['split_contamination_rate']:.1%} of test)")
    else:
        print(f"  [OK] No template families span both splits")

    print()
    if issues > 0:
        print(f"  {issues} issue(s) found. See details above.")
        print(f"  The perfect 1.0000 test-set scores may be explained by")
        print(f"  near-duplicate or template-sibling contamination.")
        print(f"\n  RECOMMENDED REMEDIATION:")
        print(f"    (a) Use group-based splitting: keep template families together.")
        print(f"    (b) Evaluate on a genuinely external test set drawn from")
        print(f"        prompts and scenario wording never used during dataset")
        print(f"        construction.")
        print(f"    (c) Apply near-duplicate removal (cosine > 0.90 or")
        print(f"        Jaccard > 0.80) before splitting.")
    else:
        print(f"  No issues found. The train/test split appears clean.")

    # Save report
    output_path = Path(json_path) if json_path else VALIDATION_DIR / "integrity_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full report saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data integrity validation for GenAI Phishing Detector"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(RAW_DATASET_CSV),
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip embedding cosine check (n-gram only)"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path for JSON output (default: validation/integrity_report.json)"
    )
    args = parser.parse_args()
    run_validation(csv_path=args.dataset, fast=args.fast, json_path=args.json)
