# Evidence Generation Guide for Project Write-Up

This guide tells you exactly what to run and screenshot for each appendix.

---

## Prerequisites

Before generating evidence, complete this order:
```
1. Train the model (Colab or local)
2. Run evaluation (evaluation/evaluate.py)
3. Launch Streamlit app (streamlit run streamlit_app/app.py)
4. Run the evidence scripts below
```

---

## Appendix 12: Figure 4.1 — Dataset Distribution Bar Chart

**Command to run:**
```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/dataset.csv')
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels = ['Legitimate\nFinancial Comm.', 'Traditional\nPhishing', 'AI-Generated\nPhishing']

plt.figure(figsize=(8, 5))
counts = df['label'].value_counts().sort_index()
bars = plt.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5)

for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=13)

plt.ylabel('Number of Samples', fontsize=12)
plt.title('Figure 4.1: Dataset Class Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('dataset_distribution.png', dpi=200, bbox_inches='tight')
plt.show()
"
```

**What to do:** Run the command → a bar chart appears → screenshot it → save as `Figure 4.1 Dataset Distribution.png`

---

## Appendix 13: Table 4.1 — Dataset Composition

**Command to run:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('dataset/dataset.csv')
counts = df['label'].value_counts().sort_index()
print('============================================')
print('Table 4.1: Dataset Composition')
print('============================================')
print(f'| Class                                     | Samples |')
print(f'|-------------------------------------------|---------|')
print(f'| Legitimate Financial Communication        | {counts[0]:>5}   |')
print(f'| Traditional Phishing                      | {counts[1]:>5}   |')
print(f'| AI-Generated Phishing                     | {counts[2]:>5}   |')
print(f'|-------------------------------------------|---------|')
print(f'| Total                                     | {len(df):>5}   |')
print('============================================')
"
```

**What to do:** Copy the output directly into your project table.

---

## Appendix 8: Table 4.3 — Hyperparameter Configuration

**Command to run:**
```bash
python -c "
from config.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, MAX_LENGTH
print('============================================')
print('Table 4.3: Hyperparameter Configuration')
print('============================================')
print(f'| Parameter         | Value                 |')
print(f'|-------------------|-----------------------|')
print(f'| Pre-trained Model | {MODEL_NAME}')
print(f'| Max Sequence Len  | {MAX_LENGTH}')
print(f'| Learning Rate     | {LEARNING_RATE}')
print(f'| Batch Size        | {BATCH_SIZE}')
print(f'| Epochs            | {EPOCHS}')
print(f'| Optimizer         | AdamW                 |')
print(f'| Loss Function     | Cross Entropy         |')
print(f'| Weight Decay      | 0.01                  |')
print(f'| Scheduler         | Linear                |')
print(f'| Warmup Steps      | 0                     |')
print('============================================')
"
```

**What to do:** Copy the output into your project table.

---

## Appendix 5: Table 4.4 — Model Performance

**Command to run:**
```bash
python -c "
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
from evaluation.evaluate import evaluate_model

metrics = evaluate_model()
print()
print('============================================')
print('Table 4.4: Model Performance')
print('============================================')
print(f'| Metric    | Value  |')
print(f'|-----------|--------|')
print(f'| Accuracy  | {metrics[\"accuracy\"]:.4f} |')
print(f'| Precision | {metrics[\"weighted_precision\"]:.4f} |')
print(f'| Recall    | {metrics[\"weighted_recall\"]:.4f} |')
print(f'| F1-Score  | {metrics[\"weighted_f1\"]:.4f} |')
print(f'| AUPRC     | (see evaluation report) |')
print('============================================')
"
```

**What to do:** After running `evaluation/evaluate.py`, the metrics are printed. Copy them into the table.

---

## Appendix 5b: Confusion Matrix Screenshot

**Location after running evaluation:**
```
evaluation/figures/confusion_matrix.png
```

**What to do:**
1. Run: `python evaluation/evaluate.py`
2. Open: `evaluation/figures/confusion_matrix.png`
3. Crop the image to show just the matrix
4. Insert into your project

---

## Appendix 3: Table 4.6 — Inference Latency

**Command to run:**
```bash
python -c "
import time
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load model
model = DistilBertForSequenceClassification.from_pretrained('models/phishing_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('models/tokenizer')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load test messages
df = pd.read_csv('dataset/dataset.csv')
texts = df['text'].sample(100, random_state=42).tolist()

# Time inference
times = []
for text in texts:
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=enc['input_ids'].to(device),
                       attention_mask=enc['attention_mask'].to(device))
    elapsed = (time.perf_counter() - start) * 1000  # milliseconds
    times.append(elapsed)

times = np.array(times)
print('============================================')
print('Table 4.6: Inference Latency')
print('============================================')
print(f'| Metric                | Value         |')
print(f'|-----------------------|---------------|')
print(f'| Average Response Time | {times.mean():.2f} ms     |')
print(f'| Fastest Response      | {times.min():.2f} ms     |')
print(f'| Slowest Response      | {times.max():.2f} ms     |')
print(f'| Samples Tested        | {len(times)}           |')
print('============================================')
"
```

**What to do:** Run the command and copy the values into your table.

---

## Appendix 4: Table 4.5 — Comparison with Regex Baseline

**Run this script:**
```bash
python -c "
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load test data
df = pd.read_csv('dataset/dataset.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = df.iloc[-156:]  # ~10%
texts = test_df['text'].tolist()
y_true = test_df['label'].values

# --- REGEX BASELINE ---
def regex_classify(text):
    t = text.lower()
    # AI phishing indicators
    ai_score = 0
    ai_patterns = ['central bank of nigeria', 'cbn directive', 'regulatory compliance',
                   'beneficial ownership', 'anti-money laundering', 'know your customer']
    for p in ai_patterns:
        if p in t: ai_score += 1

    # Traditional phishing indicators
    trad_score = 0
    trad_patterns = ['urgent', 'click here', 'suspended', 'blocked', 'verify.*account',
                    'bvn.*block', 'reactivate', 'immediate']
    for p in trad_patterns:
        if re.search(p, t): trad_score += 1

    # Legitimate indicators
    legit_score = 0
    legit_patterns = ['debit alert', 'credit alert', 'available balance',
                     'transaction alert', 'monthly statement']
    for p in legit_patterns:
        if p in t: legit_score += 1

    if ai_score > trad_score and ai_score > legit_score:
        return 2
    elif trad_score > legit_score:
        return 1
    return 0

y_pred_regex = np.array([regex_classify(t) for t in texts])

acc_regex = accuracy_score(y_true, y_pred_regex)
p_regex, r_regex, f_regex, _ = precision_recall_fscore_support(y_true, y_pred_regex, average='weighted', zero_division=0)

# --- PROPOSED MODEL (DistilBERT) ---
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained('models/phishing_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('models/tokenizer')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

all_preds = []
for text in texts:
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=enc['input_ids'].to(device),
                       attention_mask=enc['attention_mask'].to(device))
    pred = torch.argmax(outputs.logits, dim=-1).item()
    all_preds.append(pred)

y_pred_model = np.array(all_preds)

acc_model = accuracy_score(y_true, y_pred_model)
p_model, r_model, f_model, _ = precision_recall_fscore_support(y_true, y_pred_model, average='weighted', zero_division=0)

print('============================================')
print('Table 4.5: Comparison with Regex Baseline')
print('============================================')
print(f'| Metric    | Regex-Based Model | Proposed Framework |')
print(f'|-----------|------------------:|-------------------:|')
print(f'| Accuracy  | {acc_regex:.4f}             | {acc_model:.4f}              |')
print(f'| Precision | {p_regex:.4f}             | {p_model:.4f}              |')
print(f'| Recall    | {r_regex:.4f}             | {r_model:.4f}              |')
print(f'| F1-Score  | {f_regex:.4f}             | {f_model:.4f}              |')
print('============================================')
"
```

**What to do:** Run the command and copy the values into your comparison table.

---

## Appendix 2: Table 4.7 — Zero-Shot Detection Results

**Command to run:**
```bash
python -c "
from explainability.shap_explainer import explain_text

scenarios = [
    ('Scenario 1: Legitimate Transaction',
     'Dear Chidi, a debit of NGN25,000 was made on your GTBank account '
     '0123456789 at ShopRite on 15-Jan-2025. Available balance: NGN450,000.'),
    ('Scenario 2: Traditional Phishing',
     'URGENT!!! Your BVN has been BLOCKED. Click here to verify now: '
     'https://account-verify.tk/38472'),
    ('Scenario 3: AI-Generated Phishing',
     'Subject: Mandatory BVN-NIN Linkage Compliance Notice\n\n'
     'Dear Customer, the Central Bank of Nigeria (CBN) requires all accounts '
     'to have BVN linked to NIN by 30-Mar-2025. Non-compliant accounts will '
     'be restricted. Complete linkage: https://secure-gtbank-portal.com/verify'),
]

print('============================================')
print('Table 4.7: Zero-Shot Detection Results')
print('============================================')
print(f'| {\"Test Scenario\":25s} | {\"Expected\":22s} | {\"Predicted\":22s} | Status    |')
print(f'|{\"-\"*27}|{\"-\"*24}|{\"-\"*24}|{\"-\"*11}|')

for name, text in scenarios:
    result = explain_text(text)
    if 'Legitimate' in name:
        expected = 'Legitimate'
    elif 'Traditional' in name:
        expected = 'Traditional Phishing'
    else:
        expected = 'AI-Generated Phishing'

    predicted = result['prediction']
    status = 'Correct' if predicted == expected else 'Incorrect'
    print(f'| {name:25s} | {expected:22s} | {predicted:22s} | {status:9s} |')
print('============================================')
"
```

**What to do:** Run and copy the output table. If the model is not yet trained, use fallback mode results.

---

## Appendix 1: SHAP Explanation Screenshot

**Steps:**
1. Launch Streamlit: `streamlit run streamlit_app/app.py`
2. Wait for the app to load
3. Paste this AI-phishing message into the text area:
   ```
   Subject: Mandatory BVN-NIN Linkage Compliance Notice

   Dear Chidi Okonkwo,

   This is to notify you that the Central Bank of Nigeria (CBN) now requires
   all bank accounts to have their BVN linked to the National Identification
   Number (NIN) by 30-Mar-2025. Accounts not complying will be placed on
   restricted status.

   To complete the linkage securely, please visit:
   https://secure.gtbank-portal.com/verify-3847

   Thank you for your cooperation.
   Compliance Department
   GTBank
   ```
4. Click **Analyze**
5. Wait for SHAP explanation to load (~10-30 seconds)
6. Screenshot the **SHAP Explanation** section (the natural language text)
7. Also screenshot the **Word Importance** bar chart
8. Crop and insert both into your project

---

## Appendix 6: Listing 4.3 — SHAP Integration Code

**File to open:** `explainability/shap_explainer.py`

**Lines to screenshot/copy (key sections):**

```python
# The PhishingExplainer class:
# Lines 40-80: __init__ method (model loading)
# Lines 83-110: _predict_proba (wrapper for SHAP)
# Lines 113-140: _build_explainer (SHAP explainer creation)
# Lines 143-200: explain method (full explanation pipeline)
# Lines 203-225: _generate_explanation (natural language output)
```

**What to do:** Open `explainability/shap_explainer.py` in any text editor. The key code sections are:
- The `__init__` method (model + tokenizer loading)
- The `_predict_proba` method (prediction wrapper)
- The `_build_explainer` method (SHAP initialization)
- The `explain` method (full pipeline)
- The `_generate_explanation` method (text output)

Copy these into your appendix labeled "Listing 4.3: SHAP Integration Code".

---

## Appendix 7: Figure 4.3 — Training Loss Curve & Figure 4.4 — Validation Accuracy Curve

**Run this AFTER training:**
```bash
python -c "
import matplotlib.pyplot as plt
import numpy as np

# REPLACE these values with your actual training output
# (Copy from the Colab training output or training logs)
epochs = [1, 2, 3, 4, 5]
training_loss = [0.8234, 0.4532, 0.2876, 0.1987, 0.1456]    # <- REPLACE with YOUR values
val_accuracy = [0.7215, 0.8651, 0.9213, 0.9438, 0.9562]     # <- REPLACE with YOUR values
val_f1 = [0.7198, 0.8637, 0.9205, 0.9429, 0.9556]            # <- REPLACE with YOUR values

# Figure 4.3: Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, training_loss, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Figure 4.3: Training Loss Curve', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig('training_loss_curve.png', dpi=200, bbox_inches='tight')
plt.show()

# Figure 4.4: Validation Accuracy + F1
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accuracy, 'g-s', linewidth=2, markersize=8, label='Accuracy')
plt.plot(epochs, val_f1, 'm-d', linewidth=2, markersize=8, label='F1 Score')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Figure 4.4: Validation Accuracy & F1 Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xticks(epochs)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('validation_accuracy_curve.png', dpi=200, bbox_inches='tight')
plt.show()
"
```

**IMPORTANT:** Replace the placeholder values (`training_loss`, `val_accuracy`, `val_f1`) with the actual values printed during your Colab training. The output looks like:
```
Epoch 1: loss=0.8234, accuracy=0.7215, ..., f1=0.7198
Epoch 2: loss=0.4532, accuracy=0.8651, ..., f1=0.8637
...
```

**What to do:**
1. Copy the loss/accuracy/f1 values from your Colab training output
2. Paste them into the command above
3. Run it → two graphs appear → screenshot both

---

## Appendix 9: Listing 4.2 — DistilBERT Fine-Tuning Code

**File to open:** `training/train.py`

**Key sections to copy:**
```python
# Lines 90-110: load_and_split_dataset function
# Lines 113-137: tokenize_dataset function
# Lines 140-175: model loading (DistilBertForSequenceClassification)
# Lines 178-215: TrainingArguments configuration
# Lines 218-240: Trainer initialization
# Lines 243-250: train() and save_model()
```

**What to do:** Open `training/train.py` and copy these sections into your appendix labeled "Listing 4.2: DistilBERT Fine-Tuning".

---

## Appendix 10: Figure 4.2 — Data Preprocessing Pipeline

**What to draw:**

Create a simple flowchart (using any tool: draw.io, PowerPoint, or even hand-drawn) showing:

```
Raw Dataset (dataset.csv)
       |
       v
[1] Text Cleaning
    - Remove HTML tags
    - Remove URLs (-> [URL])
    - Remove emails (-> [EMAIL])
    - Remove special characters
       |
       v
[2] Text Normalization
    - Lowercase
    - Collapse whitespace
       |
       v
[3] Duplicate Removal
    - Drop duplicate texts
       |
       v
[4] Tokenization (DistilBERT)
    - max_length=128
    - Truncation + Padding
       |
       v
[5] Label Encoding
    - 0 = Legitimate
    - 1 = Traditional Phishing
    - 2 = AI-Generated Phishing
       |
       v
[6] Train/Validation/Test Split
    - 80% / 10% / 10%
```

**What to do:** Draw this as a simple flow diagram and insert as "Figure 4.2: Data Preprocessing Pipeline".

---

## Appendix 11: Listing 4.1 — Data Preprocessing Code

**File to open:** `preprocessing/preprocessing.py`

**Key sections to copy:**
```python
# Lines 35-50: clean_text function
# Lines 53-65: normalize_text function
# Lines 68-80: remove_duplicates function
# Lines 83-120: TextPreprocessor class (tokenize_data, fit_label_encoder)
# Lines 145-175: preprocess_pipeline (end-to-end)
# Lines 178-195: preprocess_single (inference)
```

**What to do:** Open `preprocessing/preprocessing.py` and copy these sections into your appendix labeled "Listing 4.1: Data Preprocessing Code".

---

## Quick Reference: Where Everything Lives

| Appendix | Content | Source File / Command |
|----------|---------|-----------------------|
| 1 | SHAP screenshot | `streamlit_app/app.py` (run and screenshot) |
| 2 | Zero-shot results | Run the zero-shot command above |
| 3 | Inference latency | Run the latency command above |
| 4 | Regex comparison | Run the comparison command above |
| 5a | Model performance table | `evaluation/evaluate.py` output |
| 5b | Confusion matrix | `evaluation/figures/confusion_matrix.png` |
| 6 | SHAP code | `explainability/shap_explainer.py` |
| 7 | Training curves | Run the curve plotting command above |
| 8 | Hyperparameters | `config/config.py` |
| 9 | Fine-tuning code | `training/train.py` |
| 10 | Pipeline diagram | Draw from the specification above |
| 11 | Preprocessing code | `preprocessing/preprocessing.py` |
| 12 | Dataset distribution | Run the bar chart command above |
| 13 | Dataset composition | Run the table command above |

---

## One Command to Generate Everything (after training)

```bash
# Generate ALL tables and figures at once:
python evaluation/evaluate.py                                          # produces confusion_matrix.png + report
python -c "import pandas as pd; df=pd.read_csv('dataset/dataset.csv'); print(df['label'].value_counts().sort_index())"  # class counts
streamlit run streamlit_app/app.py                                     # for SHAP screenshot (manual)
# Then run each of the appendix commands in this guide
```

**Order of operations for minimum friction:**
1. Train in Colab → download model
2. Place model in `models/phishing_model/` and `models/tokenizer/`
3. Run `python evaluation/evaluate.py` → generates confusion matrix + metrics
4. Run the latency script → get Table 4.6 values
5. Run the regex comparison → get Table 4.5 values
6. Run the zero-shot script → get Table 4.7 values
7. Run the training curves script → get Figure 4.3 & 4.4
8. Run the bar chart script → get Figure 4.1
9. Launch Streamlit → screenshot SHAP explanation → get Appendix 1
10. Open source files → copy code listings → get Appendix 6, 9, 11
