"""
generate_evidence.py — Generate ALL appendices for project write-up
Run this in Colab or any environment with the trained model.
"""
import os, sys, json, time, re, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score,
                             classification_report)
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import shap

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

# ── 0. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('dataset/dataset.csv')
label_map = {0: 'Legitimate Financial Communication',
             1: 'Traditional Phishing',
             2: 'AI-Generated Phishing'}
df['class_name'] = df['label'].map(label_map)
counts = df['label'].value_counts().sort_index()

# Split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(df)); val_size = int(0.1 * len(df))
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]
texts, y_true = test_df['text'].tolist(), test_df['label'].values

# ── 1. Load Model ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model_path = 'models/phishing_model'
tokenizer_path = 'models/tokenizer'
if not os.path.exists(model_path):
    with zipfile.ZipFile('models/phishing_model.zip', 'r') as z:
        z.extractall('models/')
if not os.path.exists(tokenizer_path):
    with zipfile.ZipFile('models/tokenizer.zip', 'r') as z:
        z.extractall('models/')

tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# ── 2. Evaluate ───────────────────────────────────────────────────────────────
print('\n=== Evaluating on test set ===')
all_preds, all_probs = [], []
for i in range(0, len(texts), 32):
    batch = texts[i:i+32]
    enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1)
    all_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    all_probs.extend(probs.cpu().numpy())

y_pred = np.array(all_preds)
y_prob = np.array(all_probs)
acc = accuracy_score(y_true, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
auprc = np.mean([average_precision_score((y_true == i).astype(int), y_prob[:, i]) for i in range(3)])

# ── 3. Appendix 13: Dataset Composition ───────────────────────────────────────
print('\n' + '='*55)
print('TABLE 4.1: DATASET COMPOSITION')
print('='*55)
print(f'{"Class":40s} {"Samples":>8s}')
print('-'*55)
for i in range(3):
    print(f'{label_map[i]:40s} {counts[i]:>8d}')
print('-'*55)
print(f'{"Total":40s} {len(df):>8d}')
print('='*55)

# ── 4. Appendix 12: Dataset Distribution ──────────────────────────────────────
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels_c = ['Legitimate\nFinancial Comm.', 'Traditional\nPhishing', 'AI-Generated\nPhishing']
plt.figure(figsize=(9, 5))
bars = plt.bar(labels_c, counts.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=13)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Figure 4.1: Dataset Class Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/dataset_distribution.png', dpi=200, bbox_inches='tight')
print('\nSaved: figures/dataset_distribution.png')

# ── 5. Appendix 5a: Model Performance ─────────────────────────────────────────
print('\n' + '='*55)
print('TABLE 4.4: MODEL PERFORMANCE')
print('='*55)
print(f'{"Metric":25s} {"Result":>12s}')
print('-'*55)
print(f'{"Accuracy":25s} {acc:>12.4f}')
print(f'{"Precision (weighted)":25s} {p:>12.4f}')
print(f'{"Recall (weighted)":25s} {r:>12.4f}')
print(f'{"F1-Score (weighted)":25s} {f:>12.4f}')
print(f'{"AUPRC (macro)":25s} {auprc:>12.4f}')
print('='*55)
print('\nPer-class report:')
print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Trad Phishing', 'AI Phishing']))

# ── 6. Appendix 5b: Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Trad Phishing', 'AI Phishing'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix — DistilBERT Phishing Detector', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=200, bbox_inches='tight')
print('\nSaved: figures/confusion_matrix.png')

# ── 7. Appendix 3: Inference Latency ──────────────────────────────────────────
times_ms = []
sample_texts = test_df['text'].sample(100, random_state=42).tolist()
for text in sample_texts:
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(**enc)
    times_ms.append((time.perf_counter() - start) * 1000)

times = np.array(times_ms)
print('\n' + '='*55)
print('TABLE 4.6: INFERENCE LATENCY')
print('='*55)
print(f'{"Metric":25s} {"Value":>15s}')
print('-'*55)
print(f'{"Average Response Time":25s} {times.mean():>10.2f} ms')
print(f'{"Fastest Response":25s} {times.min():>10.2f} ms')
print(f'{"Slowest Response":25s} {times.max():>10.2f} ms')
print(f'{"Std Deviation":25s} {times.std():>10.2f} ms')
print(f'{"Samples Tested":25s} {len(times):>10d}')
print('='*55)

# ── 8. Appendix 4: Regex Comparison ───────────────────────────────────────────
def regex_classify(text):
    t = text.lower()
    ai_pats = ['central bank of nigeria', 'cbn directive', 'regulatory compliance',
               'beneficial ownership', 'anti-money laundering', 'know your customer',
               'aml', 'kyc', 'compliance notice', 'mandatory']
    trad_pats = ['urgent', 'click here', 'suspended', 'blocked', 'verify.*account',
                 'bvn.*block', 'reactivate', 'immediate', 'warning', 'expired']
    legit_pats = ['debit alert', 'credit alert', 'available balance',
                  'transaction alert', 'monthly statement', 'account balance',
                  'deposit', 'withdrawal']
    ai_s = sum(1 for p in ai_pats if p in t)
    trad_s = sum(1 for p in trad_pats if re.search(p, t))
    legit_s = sum(1 for p in legit_pats if p in t)
    if ai_s > trad_s and ai_s > legit_s: return 2
    elif trad_s > legit_s: return 1
    return 0

y_pred_regex = np.array([regex_classify(t) for t in texts])
acc_r = accuracy_score(y_true, y_pred_regex)
p_r, r_r, f_r, _ = precision_recall_fscore_support(y_true, y_pred_regex, average='weighted', zero_division=0)

print('\n' + '='*70)
print('TABLE 4.5: COMPARISON WITH REGEX BASELINE')
print('='*70)
print(f'{"Metric":12s} {"Regex-Based":>16s} {"Proposed":>16s} {"Improvement":>14s}')
print('-'*70)
print(f'{"Accuracy":12s} {acc_r:>10.4f}      {acc:>10.4f}      {(acc-acc_r):>+8.4f}')
print(f'{"Precision":12s} {p_r:>10.4f}      {p:>10.4f}      {(p-p_r):>+8.4f}')
print(f'{"Recall":12s} {r_r:>10.4f}      {r:>10.4f}      {(r-r_r):>+8.4f}')
print(f'{"F1-Score":12s} {f_r:>10.4f}      {f:>10.4f}      {(f-f_r):>+8.4f}')
print('='*70)

# ── 9. Appendix 2: Zero-Shot ──────────────────────────────────────────────────
scenarios = [
    ('Scenario 1: Legitimate Transaction',
     'Dear Chidi, a debit of NGN25,000 was made on your GTBank account '
     '0123456789 at ShopRite on 15-Jan-2025. Available balance: NGN450,000.'),
    ('Scenario 2: Traditional Phishing',
     'URGENT!!! Your BVN has been BLOCKED! Click here to verify now: '
     'https://account-verify.tk/38472'),
    ('Scenario 3: AI-Generated Phishing',
     'Subject: Mandatory BVN-NIN Linkage Compliance Notice\n\n'
     'Dear Customer, the Central Bank of Nigeria (CBN) requires all accounts '
     'to have BVN linked to NIN by 30-Mar-2025. Non-compliant accounts will '
     'be restricted. Complete linkage: https://secure-gtbank-portal.com/verify'),
]
expected = ['Legitimate', 'Traditional Phishing', 'AI-Generated Phishing']

print('\n' + '='*95)
print('TABLE 4.7: ZERO-SHOT DETECTION RESULTS')
print('='*95)
print(f'{"Test Scenario":30s} {"Expected":25s} {"Predicted":25s} {"Status":>8s}')
print('-'*95)
for i, (name, text) in enumerate(scenarios):
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    status = 'Correct' if pred == i else 'Incorrect'
    print(f'{name:30s} {expected[i]:25s} {expected[pred]:25s} {status:>8s}')
print('='*95)

# ── 10. Appendix 8: Hyperparameters ───────────────────────────────────────────
try:
    from config.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, MAX_LENGTH
except:
    EPOCHS=5; BATCH_SIZE=16; LEARNING_RATE=2e-5; MODEL_NAME='distilbert-base-uncased'; MAX_LENGTH=128

print('\n' + '='*55)
print('TABLE 4.3: HYPERPARAMETER CONFIGURATION')
print('='*55)
print(f'{"Parameter":25s} {"Value":>25s}')
print('-'*55)
print(f'{"Pre-trained Model":25s} {MODEL_NAME:>25s}')
print(f'{"Max Sequence Length":25s} {MAX_LENGTH:>25d}')
print(f'{"Learning Rate":25s} {LEARNING_RATE:>25.0e}')
print(f'{"Batch Size":25s} {BATCH_SIZE:>25d}')
print(f'{"Epochs":25s} {EPOCHS:>25d}')
print(f'{"Optimizer":25s} {"AdamW":>25s}')
print(f'{"Loss Function":25s} {"Cross Entropy":>25s}')
print(f'{"Weight Decay":25s} {"0.01":>25s}')
print('='*55)

# ── 11. SHAP Explanation ──────────────────────────────────────────────────────
test_text = (
    'Subject: Mandatory BVN-NIN Linkage Compliance Notice\n\n'
    'Dear Chidi Okonkwo,\n\n'
    'This is to notify you that the Central Bank of Nigeria (CBN) now requires '
    'all bank accounts to have their BVN linked to the National Identification '
    'Number (NIN) by 30-Mar-2025. Accounts not complying will be placed on '
    'restricted status.\n\n'
    'To complete the linkage securely, please visit: '
    'https://secure.gtbank-portal.com/verify-3847\n\n'
    'Thank you for your cooperation.\n'
    'Compliance Department\n'
    'GTBank'
)

enc = tokenizer(test_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
enc = {k: v.to(device) for k, v in enc.items()}
with torch.no_grad():
    outputs = model(**enc)
pred_class = torch.argmax(outputs.logits, dim=-1).item()
probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

label_names = ['Legitimate', 'Traditional Phishing', 'AI-Generated Phishing']
print(f'\n=== SHAP Example ===')
print(f'Prediction: {label_names[pred_class]}  (confidence: {probs[pred_class]:.4f})')

class SHAPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.model(input_ids=input_ids.long(), attention_mask=attention_mask).logits

shap_model = SHAPWrapper(model).to(device)
explainer = shap.Explainer(shap_model, tokenizer, output_names=label_names, seed=42)
shap_values = explainer([test_text], max_evals=50, batch_size=1)

for class_idx in range(3):
    plt.figure()
    shap.waterfall_plot(shap_values[0, :, class_idx], show=False, max_display=15)
    plt.title(f'SHAP Waterfall — {label_names[class_idx]}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figures/shap_waterfall_{class_idx}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: figures/shap_waterfall_{class_idx}.png')

# ── 12. Package everything ────────────────────────────────────────────────────
print('\n=== All generated files ===')
for f in sorted(Path('figures').glob('*.png')):
    print(f'  {f.name} ({f.stat().st_size / 1024:.1f} KB)')

with zipfile.ZipFile('evidence_figures.zip', 'w') as z:
    for f in Path('figures').glob('*.png'):
        z.write(f, arcname=f.name)
print(f'\nAll figures zipped: evidence_figures.zip')
print('\nDone! Open each PNG, crop as needed, and insert into your write-up.')
