"""
Group-Based (Template-Aware) Splitting for GenAI Phishing Detector.

Ensures no template family appears in more than one of train/val/test.
Implements Phase 3-5 of the academic review remediation.

Usage:
    python experiments/template_aware_split/split_and_validate.py
"""

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "template_aware_split"
DATA_DIR = EXPERIMENT_DIR / "data"
REPORT_DIR = EXPERIMENT_DIR / "reports"


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the metadata CSV with template_family_id."""
    df = pd.read_csv(csv_path)
    required = {"sample_id", "text", "label", "template_family_id"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Metadata CSV must have columns {required}. "
            f"Found: {set(df.columns)}. "
            f"Regenerate dataset with updated generate_dataset.py."
        )
    df["label"] = df["label"].astype(int)
    return df


def group_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset by template families (groups), not individual samples.

    NO template family may appear in more than one split.

    Algorithm:
      1. For each class, collect the set of template families.
      2. Shuffle template families deterministically.
      3. Allocate families to train/val/test approximately respecting
         the target ratios while keeping group integrity.
      4. Assign all samples from a family to that family's split.

    Args:
        df: DataFrame with columns [sample_id, text, label, template_family_id].
        train_ratio: Target fraction for training.
        val_ratio: Target fraction for validation.
        test_ratio: Target fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'validation', 'test' DataFrames.
    """
    rng = random.Random(seed)

    # Group template families by class
    class_families: Dict[int, List[str]] = defaultdict(list)
    family_sizes: Dict[str, int] = {}
    for tid, grp in df.groupby("template_family_id"):
        label = grp["label"].iloc[0]
        class_families[label].append(tid)
        family_sizes[tid] = len(grp)

    print("\nTemplate families per class:")
    for label in sorted(class_families):
        families = class_families[label]
        total_samples = sum(family_sizes[f] for f in families)
        print(f"  Class {label}: {len(families)} families, "
              f"{total_samples} samples")

    # Allocate families within each class
    train_families: Set[str] = set()
    val_families: Set[str] = set()
    test_families: Set[str] = set()

    for label in sorted(class_families):
        families = list(class_families[label])
        rng.shuffle(families)

        total = len(families)
        n_train = max(1, round(total * train_ratio))
        n_val = max(1, round(total * val_ratio))
        n_test = total - n_train - n_val

        # Ensure no split is empty
        if n_test < 1:
            n_test = 1
            n_train = total - n_val - n_test
        if n_val < 1:
            n_val = 1
            n_train = total - n_val - n_test

        train_families.update(families[:n_train])
        val_families.update(families[n_train:n_train + n_val])
        test_families.update(families[n_train + n_val:])

        print(f"  Class {label}: {n_train} train / {n_val} val / "
              f"{n_test} test families")

    # Verify no overlap
    assert not (train_families & val_families), "Train/Val overlap!"
    assert not (train_families & test_families), "Train/Test overlap!"
    assert not (val_families & test_families), "Val/Test overlap!"

    # Build splits
    def _select(families: Set[str]) -> pd.DataFrame:
        return df[df["template_family_id"].isin(families)].reset_index(drop=True)

    train_df = _select(train_families)
    val_df = _select(val_families)
    test_df = _select(test_families)

    return {"train": train_df, "validation": val_df, "test": test_df}


def validate_split(splits: Dict[str, pd.DataFrame]) -> Dict:
    """
    Validate that the group-based split has zero cross-partition families.

    Returns:
        Dictionary with validation results.
    """
    print("\n" + "=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    results = {}

    for name, df in splits.items():
        families = set(df["template_family_id"].unique())
        dist = df["label"].value_counts().sort_index().to_dict()
        results[name] = {
            "samples": len(df),
            "families": len(families),
            "family_set": families,
            "class_distribution": {int(k): int(v) for k, v in dist.items()},
        }
        print(f"\n  {name.upper()}:")
        print(f"    Samples: {len(df)}")
        print(f"    Unique template families: {len(families)}")
        for label in sorted(dist):
            from config.config import CLASS_NAMES
            cname = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
            print(f"    Class {label} ({cname}): {dist[label]}")

    # Cross-partition checks
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    overlap_found = False
    for a, b in pairs:
        shared = results[a]["family_set"] & results[b]["family_set"]
        count = len(shared)
        status = "OK" if count == 0 else "FAIL"
        if count > 0:
            overlap_found = True
        print(f"\n  {a.upper()} / {b.upper()} shared families: {count} [{status}]")
        if count > 0:
            print(f"    Shared: {sorted(shared)[:10]}")

    results["overlap_summary"] = {
        "train_val_shared": len(
            results["train"]["family_set"] & results["validation"]["family_set"]
        ),
        "train_test_shared": len(
            results["train"]["family_set"] & results["test"]["family_set"]
        ),
        "val_test_shared": len(
            results["validation"]["family_set"] & results["test"]["family_set"]
        ),
        "any_overlap": overlap_found,
    }

    if not overlap_found:
        print("\n  RESULT: Zero shared template families across all splits.")
    else:
        print("\n  RESULT: OVERLAP DETECTED — split needs correction!")

    return results


def save_splits(splits: Dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Save train/val/test CSVs (text, label format) and metadata."""
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        # Main CSV (text, label)
        csv_path = data_dir / f"{name}.csv"
        df[["text", "label"]].to_csv(csv_path, index=False)
        print(f"  Saved {name}: {csv_path} ({len(df)} samples)")

        # Metadata CSV
        meta_path = data_dir / f"{name}_metadata.csv"
        df[["sample_id", "text", "label", "template_family_id"]].to_csv(
            meta_path, index=False
        )


def save_validation_report(
    validation: Dict,
    original_results: Optional[Dict] = None,
) -> None:
    """Save validation report as JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "split_validation": {
            k: {kk: vv for kk, vv in v.items() if kk != "family_set"}
            for k, v in validation.items()
            if isinstance(v, dict)
        },
    }
    if original_results:
        report["original_random_split"] = original_results

    path = REPORT_DIR / "split_validation.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Validation report saved to: {path}")


def run_splitting(
    metadata_csv: str,
    original_csv: Optional[str] = None,
) -> Dict:
    """
    Full splitting pipeline.

    Args:
        metadata_csv: Path to dataset_metadata.csv.
        original_csv: Path to original dataset.csv (for reference).

    Returns:
        Dictionary with split DataFrames and validation results.
    """
    print("=" * 60)
    print("TEMPLATE-AWARE GROUP-BASED SPLITTING")
    print("=" * 60)

    # Load metadata
    print("\nLoading metadata ...")
    df = load_metadata(metadata_csv)
    print(f"  Total samples: {len(df)}")
    print(f"  Unique template families: {df['template_family_id'].nunique()}")
    print(f"  Class distribution: "
          f"{dict(df['label'].value_counts().sort_index())}")

    # Group-based split
    print("\nPerforming group-based split ...")
    splits = group_based_split(df)

    # Validate
    validation = validate_split(splits)

    # Save
    print("\nSaving splits ...")
    save_splits(splits, DATA_DIR)
    save_validation_report(validation)

    # Summary
    total = sum(len(v) for v in splits.values())
    print(f"\n  Total samples across splits: {total}")
    print(f"  Original dataset size: {len(df)}")
    assert total == len(df), "Sample count mismatch!"

    return {"splits": splits, "validation": validation}


if __name__ == "__main__":
    meta_path = str(PROJECT_ROOT / "dataset" / "dataset_metadata.csv")
    if not os.path.exists(meta_path):
        print(f"Metadata file not found: {meta_path}")
        print("Run: python dataset/generate_dataset.py  (with updated generator)")
        sys.exit(1)
    run_splitting(meta_path)
