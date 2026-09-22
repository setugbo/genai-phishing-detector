---
title: GenAI Phishing Detector
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.25.0
app_file: streamlit_app/app.py
pinned: false
license: mit
---

# GenAI Phishing Detector

Transformer-based semantic analysis framework for detecting AI-synthesized
phishing attacks in the Nigerian financial sector.

## Model

- **Architecture:** DistilBERT (distilbert-base-uncased)
- **Framework:** PyTorch + HuggingFace Transformers
- **Explainability:** SHAP

## Dataset

5000+ labeled samples across three classes:
- Legitimate Financial Communication
- Traditional Phishing
- AI-Generated Phishing

## Training

Train the model using Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/genai-phishing-detector/blob/main/notebooks/training_notebook.ipynb)

## Evaluation

After training, upload the model to the `models/phishing_model/` directory.
