"""
Centralised configuration for the GenAI Phishing Detector project.

All paths, hyperparameters, and model settings are defined here
to avoid hardcoded values across modules.
"""

import os
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
MODELS_DIR: Path = PROJECT_ROOT / "models"
PREPROCESSING_DIR: Path = PROJECT_ROOT / "preprocessing"
TRAINING_DIR: Path = PROJECT_ROOT / "training"
EVALUATION_DIR: Path = PROJECT_ROOT / "evaluation"
EXPLAINABILITY_DIR: Path = PROJECT_ROOT / "explainability"
STREAMLIT_DIR: Path = PROJECT_ROOT / "streamlit_app"
UTILS_DIR: Path = PROJECT_ROOT / "utils"
DEPLOYMENT_DIR: Path = PROJECT_ROOT / "deployment"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
VALIDATION_DIR: Path = PROJECT_ROOT / "validation"

# Dataset files
RAW_DATASET_CSV: Path = DATASET_DIR / "dataset.csv"
PREPROCESSED_DATASET_CSV: Path = DATASET_DIR / "preprocessed_dataset.csv"

# Model files
MODEL_SAVE_DIR: Path = MODELS_DIR / "phishing_model"
TOKENIZER_SAVE_DIR: Path = MODELS_DIR / "tokenizer"

# Evaluation artifacts
EVAL_REPORT_DIR: Path = EVALUATION_DIR / "reports"
EVAL_FIGURES_DIR: Path = EVALUATION_DIR / "figures"

# SHAP artifacts
SHAP_OUTPUT_DIR: Path = EXPLAINABILITY_DIR / "outputs"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME: str = "distilbert-base-uncased"
MAX_LENGTH: int = 128
NUM_CLASSES: int = 3
CLASS_NAMES: List[str] = ["Legitimate", "Traditional Phishing", "AI-Generated Phishing"]
CLASS_MAPPING: dict = {0: "Legitimate", 1: "Traditional Phishing", 2: "AI-Generated Phishing"}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS: int = 5
BATCH_SIZE: int = 16
LEARNING_RATE: float = 2e-5
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1
RANDOM_SEED: int = 42
WEIGHT_DECAY: float = 0.01

# ---------------------------------------------------------------------------
# Training optimisations
# ---------------------------------------------------------------------------
USE_FP16: bool = True  # mixed precision
GRADIENT_ACCUMULATION_STEPS: int = 1
WARMUP_STEPS: int = 0
LOGGING_STEPS: int = 10
SAVE_STRATEGY: str = "epoch"
EVALUATION_STRATEGY: str = "epoch"
LOAD_BEST_MODEL_AT_END: bool = True
METRIC_FOR_BEST_MODEL: str = "f1"

# ---------------------------------------------------------------------------
# SHAP configuration
# ---------------------------------------------------------------------------
SHAP_MAX_EVALS: int = 100
SHAP_BACKGROUND_SIZE: int = 50

# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------
STREAMLIT_TITLE: str = "GenAI Phishing Detector"
STREAMLIT_ICON: str = "🛡️"
STREAMLIT_LAYOUT: str = "wide"

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
_DIRS: List[Path] = [
    DATASET_DIR,
    MODELS_DIR,
    MODEL_SAVE_DIR,
    TOKENIZER_SAVE_DIR,
    EVALUATION_DIR,
    EVAL_REPORT_DIR,
    EVAL_FIGURES_DIR,
    EXPLAINABILITY_DIR,
    SHAP_OUTPUT_DIR,
    DEPLOYMENT_DIR,
    VALIDATION_DIR,
]

for _d in _DIRS:
    _d.mkdir(parents=True, exist_ok=True)
