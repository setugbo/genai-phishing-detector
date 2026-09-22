"""
Text Preprocessing Module for GenAI Phishing Detector.

Provides:
    - Text cleaning (HTML, URLs, special chars)
    - Text normalization (lowercasing, whitespace)
    - Duplicate removal
    - DistilBERT tokenization
    - Label encoding
"""

import re
import html
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer, DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")  # mask long numbers
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9\s.,!?;:'\"()-]")


def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML, URLs, emails, and special characters.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub("[URL]", text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = NON_ALPHANUMERIC.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, strip, collapse whitespace.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    text = text.lower()
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def remove_duplicates(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    """
    Remove exact-string duplicate rows based on a specified column.

    NOTE: This performs exact-string deduplication only. It does NOT detect
    semantic or near-duplicate samples (e.g., template-sibling AI-generated
    phishing messages that differ only in slot values such as names, dates,
    or amounts). Near-duplicate detection is available separately via
    validation/data_integrity.py.

    Args:
        df: Input DataFrame.
        column: Column name to check for duplicates.

    Returns:
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[column], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Removed {before - after} exact-string duplicate rows.")
    return df


class TextPreprocessor:
    """
    End-to-end text preprocessor for DistilBERT-based classification.

    Usage:
        preprocessor = TextPreprocessor(model_name="distilbert-base-uncased")
        encodings = preprocessor(texts)
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128) -> None:
        """
        Initialize the preprocessor with a DistilBERT tokenizer.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum tokenization length (truncation/padding).
        """
        self.tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(model_name)
        self.max_length: int = max_length
        self._label_encoder: SkLabelEncoder = SkLabelEncoder()

    def tokenize_data(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize a list of texts using DistilBERT tokenizer.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary with 'input_ids', 'attention_mask' as numpy arrays.
        """
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def fit_label_encoder(self, labels: List[int]) -> np.ndarray:
        """
        Fit label encoder and transform labels.

        Args:
            labels: List of integer labels (0, 1, 2).

        Returns:
            Numpy array of encoded labels.
        """
        return self._label_encoder.fit_transform(labels)

    def transform_labels(self, labels: List[int]) -> np.ndarray:
        """
        Transform labels using fitted encoder.

        Args:
            labels: List of integer labels.

        Returns:
            Numpy array of encoded labels.
        """
        return self._label_encoder.transform(labels)

    def inverse_transform_labels(self, encoded: np.ndarray) -> List[int]:
        """
        Convert encoded labels back to original values.

        Args:
            encoded: Numpy array of encoded labels.

        Returns:
            List of original label values.
        """
        return self._label_encoder.inverse_transform(encoded).tolist()

    @property
    def label_mapping(self) -> Dict[int, int]:
        """Return mapping from encoded label to original label."""
        mapping = {}
        for i, cls in enumerate(self._label_encoder.classes_):
            mapping[i] = int(cls)
        return mapping

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: Optional[str] = "label",
        deduplicate: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Run full preprocessing pipeline on a DataFrame.

        Steps:
            1. Remove duplicates (optional)
            2. Clean text
            3. Normalize text
            4. Tokenize
            5. Encode labels (if label column present)

        Args:
            df: Input DataFrame.
            text_column: Name of text column.
            label_column: Name of label column (None for inference).
            deduplicate: Whether to remove duplicate texts.

        Returns:
            Tuple of (encodings_dict, labels_array_or_None).
        """
        df = df.copy()

        if deduplicate:
            df = remove_duplicates(df, column=text_column)

        texts = df[text_column].tolist()
        texts = [clean_text(t) for t in texts]
        texts = [normalize_text(t) for t in texts]
        df[text_column] = texts

        encodings = self.tokenize_data(texts)

        labels: Optional[np.ndarray] = None
        if label_column and label_column in df.columns:
            raw_labels = df[label_column].astype(int).tolist()
            labels = self.fit_label_encoder(raw_labels)

        return encodings, labels

    def preprocess_single(self, text: str) -> Dict[str, np.ndarray]:
        """
        Preprocess a single text string for inference.

        Args:
            text: Raw input text.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        text = clean_text(text)
        text = normalize_text(text)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def create_preprocessed_dataset(
    input_csv: str,
    output_csv: str,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
) -> pd.DataFrame:
    """
    Convenience function: load CSV, preprocess, save preprocessed CSV.

    Args:
        input_csv: Path to raw dataset CSV.
        output_csv: Path to save preprocessed CSV.
        model_name: DistilBERT model variant.
        max_length: Max token length.

    Returns:
        Preprocessed DataFrame.
    """
    df = pd.read_csv(input_csv)
    preprocessor = TextPreprocessor(model_name=model_name, max_length=max_length)
    encodings, labels = preprocessor.preprocess_pipeline(df)

    # Build a DataFrame with preprocessed text + encodings
    texts = df["text"].tolist()
    texts = [normalize_text(clean_text(t)) for t in texts]
    result_df = pd.DataFrame({
        "text": texts,
        "label": labels.tolist() if labels is not None else [],
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
    })
    result_df.to_csv(output_csv, index=False)
    print(f"Preprocessed dataset saved to: {output_csv}")
    return result_df


if __name__ == "__main__":
    # Quick test
    input_path = "dataset/dataset.csv"
    output_path = "dataset/preprocessed_dataset.csv"
    create_preprocessed_dataset(input_path, output_path)
