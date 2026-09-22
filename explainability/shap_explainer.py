"""
SHAP Explainability Module for GenAI Phishing Detector.

Provides:
    - Word-level feature attribution via SHAP
    - Top-K suspicious word identification
    - Natural language explanation generation
    - Visualization data for Streamlit integration
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import shap
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    MODEL_SAVE_DIR,
    TOKENIZER_SAVE_DIR,
    CLASS_NAMES,
    CLASS_MAPPING,
    MAX_LENGTH,
    SHAP_MAX_EVALS,
    SHAP_BACKGROUND_SIZE,
    SHAP_OUTPUT_DIR,
)
from preprocessing.preprocessing import clean_text, normalize_text


# ---------------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------------

class PhishingExplainer:
    """
    SHAP-based explainer for DistilBERT phishing detection model.

    Provides word-level attributions and natural language explanations
    for why a given text was classified as legitimate, traditional phishing,
    or AI-generated phishing.

    Usage:
        explainer = PhishingExplainer()
        result = explainer.explain("Your account has been suspended...")
        print(result["explanation"])
    """

    def __init__(
        self,
        model_path: str = str(MODEL_SAVE_DIR),
        tokenizer_path: str = str(TOKENIZER_SAVE_DIR),
        max_evals: int = SHAP_MAX_EVALS,
        background_size: int = SHAP_BACKGROUND_SIZE,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise the SHAP explainer.

        Args:
            model_path: Path to fine-tuned DistilBERT model.
            tokenizer_path: Path to DistilBERT tokenizer.
            max_evals: Maximum number of SHAP evaluations.
            background_size: Number of background samples for explainer.
            device: Device to run on ('cpu', 'cuda', or None for auto-detect).
        """
        self.max_evals = max_evals
        self.background_size = background_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {model_path} ...")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded on {self.device}.")

        print(f"Loading tokenizer from {tokenizer_path} ...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

        self._explainer: Optional[shap.Explainer] = None

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for SHAP: takes list of strings, returns class probabilities.

        Args:
            texts: List of raw text strings.

        Returns:
            numpy array of shape (n_texts, n_classes) with predicted probabilities.
        """
        cleaned = [normalize_text(clean_text(t)) for t in texts]

        encodings = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        return probs

    def _build_explainer(self) -> shap.Explainer:
        """
        Build or retrieve the cached SHAP explainer.

        Returns:
            Configured SHAP Explainer instance.
        """
        if self._explainer is not None:
            return self._explainer

        # Create a masker that tokenizes text for the partition explainer
        masker = shap.maskers.Text(
            self.tokenizer,
            mask_token="...",
            collapse_mask_token=True,
        )

        self._explainer = shap.Explainer(
            self._predict_proba,
            masker=masker,
            algorithm="partition",
            output_names=CLASS_NAMES,
            max_evals=self.max_evals,
        )

        return self._explainer

    def explain(
        self,
        text: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Explain a single text sample using SHAP.

        Args:
            text: Raw input text to explain.
            top_k: Number of top contributing words to return.

        Returns:
            Dictionary containing:
                - text: The cleaned input text.
                - prediction: Predicted class label.
                - confidence: Confidence score for the prediction.
                - probabilities: Dict mapping class name to probability.
                - shap_values: Raw SHAP values (serialisable list).
                - top_words: List of (word, importance) for top contributors.
                - explanation: Natural language explanation string.
                - suspicious_words: Words flagged as suspicious indicators.
        """
        # Preprocess
        cleaned = normalize_text(clean_text(text))

        # Get prediction
        probs = self._predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = CLASS_MAPPING[pred_idx]
        probabilities = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

        # SHAP explanation
        explainer = self._build_explainer()
        shap_values = explainer([cleaned])

        # Extract SHAP values for the predicted class
        sv = shap_values[0, :, pred_idx]

        # Build word-importance pairs
        tokens = sv.data
        values = sv.values

        word_importances: List[Tuple[str, float]] = []
        for token, val in zip(tokens, values):
            token_stripped = token.strip()
            if token_stripped and token_stripped not in ("...", "[URL]", "[EMAIL]"):
                word_importances.append((token_stripped, float(val)))

        # Sort by absolute importance
        word_importances.sort(key=lambda x: abs(x[1]), reverse=True)

        top_words = word_importances[:top_k]

        # Suspicious words (positive importance for phishing classes)
        suspicious_words = [
            {"word": w, "importance": round(s, 4)}
            for w, s in word_importances
            if s > 0.01 and pred_idx in (1, 2)
        ][:top_k]

        # Generate explanation text
        explanation = self._generate_explanation(
            text=cleaned,
            prediction=pred_label,
            confidence=confidence,
            top_words=top_words,
            pred_idx=pred_idx,
        )

        return {
            "text": cleaned,
            "prediction": pred_label,
            "prediction_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_words": [{"word": w, "importance": round(s, 4)} for w, s in top_words],
            "suspicious_words": suspicious_words,
            "explanation": explanation,
        }

    def _generate_explanation(
        self,
        text: str,
        prediction: str,
        confidence: float,
        top_words: List[Tuple[str, float]],
        pred_idx: int,
    ) -> str:
        """
        Generate a human-readable explanation of the prediction.

        Args:
            text: Cleaned input text.
            prediction: Predicted class name.
            confidence: Prediction confidence.
            top_words: List of (word, importance) tuples.
            pred_idx: Predicted class index.

        Returns:
            Formatted explanation string.
        """
        positive_words = [w for w, s in top_words if s > 0.01][:5]

        if pred_idx == 0:
            # Legitimate
            base = (
                f"This message was classified as '{prediction}' "
                f"with {confidence:.1%} confidence. "
            )
            if positive_words:
                base += (
                    f"The words '{', '.join(positive_words)}' "
                    f"are consistent with legitimate financial communications."
                )
            else:
                base += "No suspicious patterns were detected in this message."
            return base

        reason_phrases = {
            1: {
                "trigger": "traditional phishing indicators",
                "because": "contain typical phishing patterns such as urgent requests, "
                           "poor grammar, and suspicious links",
            },
            2: {
                "trigger": "AI-generated phishing indicators",
                "because": "exhibit sophisticated social engineering techniques, "
                           "professional yet manipulative language, and "
                           "authority-based psychological persuasion",
            },
        }

        info = reason_phrases.get(pred_idx, reason_phrases[1])

        base = (
            f"This email was classified as '{prediction}' "
            f"with {confidence:.1%} confidence because it "
            f"{info['because']}."
        )

        if positive_words:
            base += (
                f"\n\nThe words '{', '.join(positive_words)}' "
                f"contributed significantly to the prediction."
            )

        return base

    def get_word_attribution(
        self,
        text: str,
    ) -> List[Dict]:
        """
        Get SHAP word attributions for a given text.

        Args:
            text: Raw input text.

        Returns:
            List of dicts with 'word', 'importance' for every token.
        """
        cleaned = normalize_text(clean_text(text))
        explainer = self._build_explainer()
        shap_values = explainer([cleaned])

        # Get prediction
        probs = self._predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(probs))

        sv = shap_values[0, :, pred_idx]
        tokens = sv.data
        values = sv.values

        attributions = []
        for token, val in zip(tokens, values):
            token_stripped = token.strip()
            if token_stripped:
                attributions.append({
                    "word": token_stripped,
                    "importance": float(val),
                })

        return attributions

    def batch_explain(
        self,
        texts: List[str],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Explain multiple texts.

        Args:
            texts: List of raw text strings.
            top_k: Number of top words per text.

        Returns:
            List of explanation dictionaries.
        """
        return [self.explain(t, top_k=top_k) for t in texts]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def explain_text(
    text: str,
    model_path: str = str(MODEL_SAVE_DIR),
    tokenizer_path: str = str(TOKENIZER_SAVE_DIR),
    top_k: int = 10,
) -> Dict:
    """
    Convenience function to explain a single text.

    Args:
        text: Raw input text.
        model_path: Path to saved model.
        tokenizer_path: Path to saved tokenizer.
        top_k: Number of top words.

    Returns:
        Explanation dictionary.
    """
    explainer = PhishingExplainer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    return explainer.explain(text, top_k=top_k)


if __name__ == "__main__":
    # Test
    sample = (
        "Your account has been restricted due to suspicious activity. "
        "Verify your identity immediately at https://secure-bank-verify.com "
        "to avoid permanent closure."
    )
    result = explain_text(sample)
    print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print(f"Explanation: {result['explanation']}")
    print(f"Top words: {result['top_words'][:5]}")
