"""
Streamlit Application for GenAI Phishing Detector.

Provides:
    - Text input (paste email/SMS/bank notification)
    - Prediction with confidence scores
    - SHAP-based word-level explainability
    - Risk level assessment
    - Color-coded results
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    STREAMLIT_TITLE,
    STREAMLIT_ICON,
    MODEL_SAVE_DIR,
    TOKENIZER_SAVE_DIR,
    CLASS_NAMES,
    CLASS_MAPPING,
)
from explainability.shap_explainer import PhishingExplainer


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon=STREAMLIT_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLORS = {
    "Legitimate": "#2ecc71",        # Green
    "Traditional Phishing": "#f39c12",  # Orange
    "AI-Generated Phishing": "#e74c3c",  # Red
}

RISK_LEVELS = {
    "Legitimate": "Low",
    "Traditional Phishing": "High",
    "AI-Generated Phishing": "Critical",
}


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_explainer() -> Optional[PhishingExplainer]:
    """Load SHAP explainer (cached to avoid reload on every interaction)."""
    model_path = str(MODEL_SAVE_DIR)
    tokenizer_path = str(TOKENIZER_SAVE_DIR)

    # Check if model exists locally
    if not (Path(model_path) / "config.json").exists():
        st.warning(
            "Trained model not found locally. "
            "Please train the model first using train.py or download from HuggingFace Hub. "
            "The app will use a fallback mode for demonstration."
        )
        return None

    try:
        explainer = PhishingExplainer(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_evals=50,  # Reduced for web app speed
            background_size=30,
        )
        return explainer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ---------------------------------------------------------------------------
# Fallback analysis (used when model is unavailable)
# ---------------------------------------------------------------------------
def _fallback_analysis(text: str) -> Dict:
    """
    Simple keyword-based fallback for demo when model is unavailable.

    Args:
        text: Input text.

    Returns:
        Explanation dict with heuristic results.
    """
    text_lower = text.lower()

    phishing_keywords = [
        "urgent", "verify", "click here", "suspended", "blocked",
        "bvn", "nin", "account restricted", "immediately", "reactivate",
        "update your account", "confirm your", "security alert",
        "unauthorised", "suspicious activity", "compliance",
        "final notice", "failure to", "permanent closure",
    ]

    ai_indicators = [
        "central bank of nigeria", "cbn directive", "regulatory compliance",
        "anti-money laundering", "beneficial ownership",
        "know your customer", "kyc update",
    ]

    legitimate_indicators = [
        "debit alert", "credit alert", "available balance",
        "transaction alert", "password reset",
        "monthly statement", "pre-qualified",
    ]

    legit_score = sum(1 for kw in legitimate_indicators if kw in text_lower)
    phish_score = sum(2 for kw in phishing_keywords if kw in text_lower)
    ai_score = sum(3 for kw in ai_indicators if kw in text_lower)

    if ai_score > phish_score and ai_score > legit_score:
        pred_idx = 2
    elif phish_score > legit_score:
        pred_idx = 1
    else:
        pred_idx = 0

    pred_label = CLASS_MAPPING[pred_idx]
    confidence = 0.65 + 0.1 * (max(legit_score, phish_score, ai_score) / max(1, len(phishing_keywords)))

    return {
        "prediction": pred_label,
        "prediction_index": pred_idx,
        "confidence": min(confidence, 0.95),
        "probabilities": {
            "Legitimate": 0.7 if pred_idx == 0 else 0.15,
            "Traditional Phishing": 0.2 if pred_idx == 1 else 0.1,
            "AI-Generated Phishing": 0.1 if pred_idx == 2 else 0.75,
        },
        "text": text,
        "top_words": [{"word": "fallback", "importance": 0.0}],
        "suspicious_words": [],
        "explanation": (
            f"This message was classified as '{pred_label}' "
            f"(demonstration mode). Train and load a model for "
            f"full SHAP-based explanations."
        ),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"# {STREAMLIT_ICON} {STREAMLIT_TITLE}")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        """
        This tool detects **AI-generated phishing communications** targeting the
        **Nigerian financial sector** using semantic analysis with DistilBERT.

        **Detection Classes:**
        - ✅ Legitimate Financial Communication
        - ⚠️ Traditional Phishing
        - 🚨 AI-Generated Phishing

        **Model:** DistilBERT (distilbert-base-uncased)
        **Framework:** PyTorch + HuggingFace Transformers
        **Explainability:** SHAP
        """
    )

    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown(
        """
        1. Paste an email, SMS, or bank notification in the text area
        2. Click **Analyze**
        3. Review the prediction and explanation
        4. Check highlighted suspicious words
        """
    )

    st.markdown("---")
    st.markdown("### Supported Institutions")
    st.markdown(
        """
        GTBank • UBA • Access Bank • Zenith Bank • Fidelity Bank
        First Bank • Moniepoint • PalmPay • Opay • Stanbic IBTC
        Ecobank • Union Bank • Wema Bank • FCMB • Kuda • Carbon
        """
    )

    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool is for research and educational purposes. "
        "Always verify suspicious communications through official channels."
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title(f"{STREAMLIT_ICON} {STREAMLIT_TITLE}")
st.markdown(
    "Detect AI-crafted phishing attacks targeting Nigerian financial institutions "
    "using transformer-based semantic analysis."
)

# Input section
col_input, col_example = st.columns([3, 1])

with col_input:
    input_text = st.text_area(
        "Paste email, SMS, or bank notification:",
        height=200,
        placeholder="Paste the message you want to analyze here...",
        key="input_text",
    )

with col_example:
    st.markdown("### Sample Messages")
    example_type = st.selectbox(
        "Load example:",
        ["", "Legitimate", "Traditional Phishing", "AI-Generated Phishing"],
    )

    if example_type == "Legitimate":
        input_text = (
            "Dear Chidi, a debit of NGN15,000 was made on your GTBank account "
            "0123456789 at ShopRite on 15-Jan-2025. Available balance: NGN234,500. "
            "If not you, call 08012345678."
        )
    elif example_type == "Traditional Phishing":
        input_text = (
            "URGENT!!! Your BVN has been BLOCKED. Click here to verify now: "
            "https://account-verify.tk/38472"
        )
    elif example_type == "AI-Generated Phishing":
        input_text = (
            "Subject: Mandatory BVN-NIN Linkage Compliance Notice\n\n"
            "Dear Chidi Okonkwo,\n\n"
            "This is to notify you that the Central Bank of Nigeria (CBN) now requires "
            "all bank accounts to have their BVN linked to the National Identification Number (NIN) "
            "by 30-Mar-2025. Accounts not complying will be placed on restricted status.\n\n"
            "To complete the linkage securely, please visit: "
            "https://secure.gtbank-portal.com/verify-3847\n\n"
            "This process takes less than 2 minutes.\n\n"
            "Thank you for your cooperation.\n"
            "Compliance Department\nGTBank"
        )

    if example_type:
        st.session_state["input_text"] = input_text
        st.rerun()

# Restore from session state
if "input_text" in st.session_state:
    input_text = st.session_state["input_text"]
    del st.session_state["input_text"]

# Analyze button
analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if analyze_clicked and input_text.strip():
    explainer = load_explainer()

    with st.spinner("Analyzing message..."):
        if explainer is not None:
            try:
                result = explainer.explain(input_text, top_k=10)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        else:
            # Fallback: simple heuristic demo when model is not available
            st.warning("Model not loaded. Running in demonstration mode with heuristic analysis.")
            result = _fallback_analysis(input_text)

    # Unpack results
    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    explanation = result["explanation"]
    top_words = result.get("top_words", [])
    suspicious_words = result.get("suspicious_words", [])

    color = COLORS.get(prediction, "#95a5a6")
    risk = RISK_LEVELS.get(prediction, "Unknown")

    # -----------------------------------------------------------------------
    # Results cards
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Analysis Results")

    row1 = st.columns(3)

    # Prediction card
    with row1[0]:
        st.markdown(
            f"""
            <div style="
                background-color: {color}20;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            ">
                <h3 style="color: {color}; margin: 0;">Prediction</h3>
                <h1 style="color: {color}; margin: 10px 0;">{prediction}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Confidence card
    with row1[1]:
        st.markdown(
            f"""
            <div style="
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            ">
                <h3 style="color: #2c3e50; margin: 0;">Confidence</h3>
                <h1 style="color: {color}; margin: 10px 0;">{confidence:.1%}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risk level card
    with row1[2]:
        st.markdown(
            f"""
            <div style="
                background-color: {color}15;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            ">
                <h3 style="color: {color}; margin: 0;">Risk Level</h3>
                <h1 style="color: {color}; margin: 10px 0;">{risk}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Probability distribution
    # -----------------------------------------------------------------------
    st.markdown("### Probability Distribution")
    prob_cols = st.columns(3)
    for i, (class_name, prob) in enumerate(probabilities.items()):
        class_color = COLORS.get(class_name, "#95a5a6")
        with prob_cols[i]:
            st.markdown(f"**{class_name}**")
            st.progress(prob, text=f"{prob:.1%}")
            st.markdown(
                f"<p style='text-align:center; color:{class_color}; "
                f"font-size:24px; font-weight:bold;'>{prob:.1%}</p>",
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # SHAP Explanation
    # -----------------------------------------------------------------------
    st.markdown("### SHAP Explanation")
    st.info(explanation)

    # Show top words
    if top_words:
        st.markdown("#### Top Influential Words")
        words_df_data = []
        for item in top_words:
            imp = item.get("importance", item.get("score", 0))
            word = item.get("word", item.get("token", ""))
            bar_color = color if imp > 0 else "#7f8c8d"
            words_df_data.append({
                "Word": word,
                "Importance": imp,
                "abs_imp": abs(imp),
            })

        if words_df_data:
            words_df = __import__("pandas").DataFrame(words_df_data)
            words_df = words_df.sort_values("abs_imp", ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(3, len(words_df) * 0.4)))
            colors_bar = [
                COLORS.get(prediction, "#e74c3c") if imp > 0 else "#95a5a6"
                for imp in words_df["Importance"]
            ]
            ax.barh(
                words_df["Word"],
                words_df["Importance"],
                color=colors_bar,
                edgecolor="white",
            )
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.set_xlabel("SHAP Value (Impact on Prediction)")
            ax.set_title("Word Importance for Prediction")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    # -----------------------------------------------------------------------
    # Suspicious words highlight
    # -----------------------------------------------------------------------
    if suspicious_words:
        st.markdown("#### 🚩 Suspicious Indicators Detected")
        sus_cols = st.columns(min(3, len(suspicious_words)))
        for idx, item in enumerate(suspicious_words):
            col_idx = idx % 3
            with sus_cols[col_idx]:
                word = item.get("word", item.get("token", ""))
                imp = item.get("importance", item.get("score", 0))
                st.markdown(
                    f"""
                    <div style="
                        background-color: {color}20;
                        border-left: 4px solid {color};
                        padding: 8px 12px;
                        margin: 4px 0;
                        border-radius: 4px;
                    ">
                        <strong>"{word}"</strong>
                        <br>
                        <small>Importance: {imp:.4f}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # Full text with highlights (for phishing)
    # -----------------------------------------------------------------------
    if prediction != "Legitimate" and suspicious_words:
        st.markdown("#### Highlighted Analysis")
        highlighted = input_text
        for item in suspicious_words[:5]:
            word = item.get("word", item.get("token", ""))
            highlighted = highlighted.replace(
                word,
                f"<span style='background-color: {color}40; "
                f"padding: 2px 4px; border-radius: 3px;'>{word}</span>",
            )
        st.markdown(
            f'<div style="padding: 15px; border: 1px solid #dee2e6; '
            f'border-radius: 8px; line-height: 1.8;">{highlighted}</div>',
            unsafe_allow_html=True,
        )

elif analyze_clicked and not input_text.strip():
    st.warning("Please enter a message to analyze.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; font-size: 12px;">
        GenAI Phishing Detector — Research Project |
        Built with PyTorch, HuggingFace Transformers, SHAP & Streamlit |
        Nigerian Financial Sector Focus
    </div>
    """,
    unsafe_allow_html=True,
)
