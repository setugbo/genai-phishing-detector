"""
HuggingFace Spaces deployment setup script for GenAI Phishing Detector.

This script prepares the repository for deployment on HuggingFace Spaces
by creating the necessary configuration files and verifying the setup.

Usage:
    python deployment/setup_hf_spaces.py

For manual deployment:
    1. Create a new Space at https://huggingface.co/new-space
    2. Choose "Streamlit" as the SDK
    3. Connect your GitHub repo or upload files directly
    4. The Space will auto-detect requirements.txt and streamlit_app/app.py
"""

import os
import sys
from pathlib import Path


def verify_deployment() -> None:
    """Verify that all files required for HuggingFace Spaces are in place."""
    root = Path(__file__).resolve().parent.parent

    required_files = {
        "requirements.txt": root / "requirements.txt",
        "packages.txt": root / "packages.txt",
        "Streamlit app": root / "streamlit_app" / "app.py",
        "Config": root / "config" / "config.py",
        "Preprocessing": root / "preprocessing" / "preprocessing.py",
        "SHAP explainer": root / "explainability" / "shap_explainer.py",
    }

    print("=" * 60)
    print("GenAI Phishing Detector - HuggingFace Spaces Deployment Check")
    print("=" * 60)

    all_ok = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path.name if exists else 'MISSING'}")
        if not exists:
            all_ok = False

    print("---")

    if all_ok:
        print("\n✅ All required files are present. Ready for deployment.")
        print("\nDeployment instructions:")
        print("  1. Go to https://huggingface.co/new-space")
        print("  2. Set Space name: 'genai-phishing-detector'")
        print("  3. Choose SDK: Streamlit")
        print("  4. Hardware: CPU basic (free)")
        print("  5. Upload or connect repo")
        print("  6. Space will auto-build from requirements.txt")
        print("\nAlternatively, use the HuggingFace CLI:")
        print("  huggingface-cli login")
        print("  huggingface-cli repo create genai-phishing-detector --type space")
    else:
        print("\n❌ Some files are missing. Please ensure all modules are generated.")
        sys.exit(1)


def create_space_readme() -> None:
    """Generate the HuggingFace Space metadata README."""
    content = """---
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
"""
    readme_path = Path(__file__).resolve().parent.parent / "deployment" / "SPACE_README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"Space README created at: {readme_path}")


if __name__ == "__main__":
    verify_deployment()
    create_space_readme()
