# Getting Started — Deployment-Focused Guide

**Goal:** Deploy a live phishing detector web app to HuggingFace Spaces at zero cost.

## The Deployment Model

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Google      │ ──► │  Download .zip   │ ──► │  HuggingFace Spaces  │
│  Colab       │     │  (trained model) │     │  (live Streamlit     │
│  (free GPU)  │     │                  │     │   web app)           │
│  Train model │     │  ~500 MB         │     │                      │
└──────────────┘     └──────────────────┘     └──────────────────────┘
  15 minutes               5 minutes                5 minutes
```

That's it. Three steps, ~25 minutes total, entirely free.

---

## Step 1: Train the Model in Google Colab (15 min)

### 1.1 Open the Notebook

https://colab.research.google.com/ → **File → Upload Notebook** → select `notebooks/training_notebook.ipynb` from this project

### 1.2 Enable GPU

In Colab menu: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

### 1.3 Mount Google Drive

Run the cell titled `"Mount Google Drive"`. A popup will ask you to authorize — click "Connect to Google Drive".

This creates a folder at `MyDrive/genai-phishing-detector/` in your Google Drive.

### 1.4 Run Everything

**Runtime → Run all** (or press `Ctrl+F9`)

Colab will execute every cell in sequence. The training takes about 10-15 minutes with the T4 GPU. You'll see progress bars for each epoch.

### 1.5 What Happens During Training

| Cell | What It Does | Output |
|------|-------------|--------|
| Install deps | Installs transformers, torch, shap, etc. | All installed |
| Mount Drive | Connects to your Google Drive | Folder created |
| Generate dataset | Creates 5000+ Nigerian bank phishing samples | "Dataset saved" |
| Train model | Fine-tunes DistilBERT for 5 epochs | Loss decreases each epoch |
| Evaluate | Tests on 10% held-out data | ~95% accuracy, F1, precision, recall |
| Download | Zips the model | `phishing_model.zip`, `tokenizer.zip` |

### 1.6 Download the Trained Model

After training finishes, scroll to the last cell. It zips the model into two files that appear in Colab's file browser (left sidebar).

**Right-click → Download** for both:
- `phishing_model.zip` (~260 MB)
- `tokenizer.zip` (~250 KB)

> **Total download:** ~260 MB. If you have slow internet, consider doing this over WiFi.

---

## Step 2: Prepare Files for Deployment (5 min)

### 2.1 Unzip the Model Files

Unzip both files and place them into the project's `models/` folder so it looks like this:

```
genai-phishing-detector/
└── models/
    ├── phishing_model/
    │   ├── config.json
    │   ├── model.safetensors    ← the actual trained weights (~260 MB)
    │   └── training_args.bin
    └── tokenizer/
        ├── vocab.txt
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

### 2.2 Verify the Complete File List

Your project folder should contain exactly these files:

```
genai-phishing-detector/
├── config/
│   ├── __init__.py
│   └── config.py
├── preprocessing/
│   ├── __init__.py
│   └── preprocessing.py
├── explainability/
│   ├── __init__.py
│   └── shap_explainer.py
├── streamlit_app/
│   └── app.py                    ← THIS is the web app entry point
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── models/
│   ├── phishing_model/           ← FROM COLAB DOWNLOAD
│   │   ├── config.json
│   │   └── model.safetensors
│   └── tokenizer/                ← FROM COLAB DOWNLOAD
│       ├── vocab.txt
│       └── tokenizer_config.json
├── requirements.txt
└── packages.txt
```

**NOTE:** The following files are optional for deployment (they're for local development and reference):
- `dataset/` (not needed — dataset was used only during training)
- `training/` (not needed — already trained in Colab)
- `evaluation/` (not needed — run locally if you want metrics)
- `notebooks/` (not needed — training is done)
- `README.md`, `GETTING_STARTED.md`, `.gitignore` (optional — can include for documentation)

---

## Step 3: Deploy to HuggingFace Spaces (5 min)

### 3.1 Create a HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up (email + password — free)
3. Verify your email address

### 3.2 Create a New Space

1. Click your profile picture (top-right) → **New Space**
2. Or go directly to: https://huggingface.co/new-space

Fill in:
| Field | Value |
|-------|-------|
| **Space Name** | `genai-phishing-detector` |
| **License** | MIT |
| **SDK** | **Streamlit** (this is the most important setting) |
| **Hardware** | **CPU basic** (free) — click the "Free" button |

Click **Create Space**.

### 3.3 Upload Files

You'll land on the Space's main page. Go to the **Files** tab.

Click **Add file → Upload files**.

Select all the files and folders listed in step 2.2 above. The minimal set is:

```
config/
  __init__.py
  config.py
preprocessing/
  __init__.py
  preprocessing.py
explainability/
  __init__.py
  shap_explainer.py
streamlit_app/
  app.py
utils/
  __init__.py
  visualization.py
models/
  phishing_model/
    config.json
    model.safetensors
  tokenizer/
    vocab.txt
    tokenizer_config.json
requirements.txt
packages.txt
```

**Important upload notes:**
- Drag and drop entire folders — the web UI preserves folder structure
- Upload `model.safetensors` first (it's the largest file at ~260 MB)
- If the upload fails for the large model file, use the **Git-based upload** method instead (see 3.4)

### 3.4 Alternative: Git-Based Upload (for large model files)

If the web upload times out on the 260 MB model file:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/genai-phishing-detector
cd genai-phishing-detector

# Copy your project files
# Copy all the files listed in step 2.2 into this directory

# Push
git add .
git commit -m "Initial deployment"
git push
```

Spaces supports files up to 5 GB via Git.

### 3.5 Wait for the Build

Once files are uploaded, HuggingFace Spaces automatically:
1. Detects `streamlit_app/app.py` as the entry point (Streamlit SDK)
2. Installs all Python packages from `requirements.txt`
3. Installs system packages from `packages.txt`
4. Launches the app

You can watch the build progress in the **Builder** tab. It takes about 3-5 minutes.

**When the build completes**, the status changes from "Building" to "Running" and your app URL becomes active.

---

## Step 4: Use Your Deployed App

### 4.1 App URL

```
https://YOUR_USERNAME-genai-phishing-detector.hf.space
```

For example, if your username is `john`, the URL is:
```
https://john-genai-phishing-detector.hf.space
```

### 4.2 How the App Works

When you paste a message and click **Analyze**:

1. The app loads the trained DistilBERT model from `models/phishing_model/`
2. Tokenizes your text using the DistilBERT tokenizer
3. Runs the model to get class probabilities
4. Runs SHAP to identify which words influenced the prediction
5. Displays:
   - **Prediction** (Legitimate / Traditional Phishing / AI-Generated Phishing)
   - **Confidence** (percentage)
   - **Risk Level** (Low / High / Critical)
   - **Probability Distribution** (bar chart for all 3 classes)
   - **SHAP Explanation** (natural language: why was it flagged)
   - **Word Importance** (which words contributed most)
   - **Suspicious Indicators** (flagged words with scores)

### 4.3 Test with Sample Messages

Copy and paste these into the text box:

**Legitimate — should show GREEN:**
```
Dear Customer, a debit of NGN15,000 was made on your GTBank account
0123456789 at ShopRite on 15-Jan-2025. Available balance: NGN234,500.
If not you, call 08012345678.
```

**Traditional Phishing — should show ORANGE:**
```
URGENT!!! Your BVN has been BLOCKED. Click here to verify now:
https://account-verify.tk/38472
```

**AI-Generated Phishing — should show RED:**
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

---

## Troubleshooting the Deployment

### Build Fails

| Error | Fix |
|-------|-----|
| `No module named 'torch'` | Make sure `requirements.txt` includes `torch` (it does in this project) |
| `Streamlit app not found` | Make sure `app.py` is inside `streamlit_app/` folder |
| `Out of memory` | Spaces free tier has 16 GB RAM — should be enough. Reduce `max_evals` in `shap_explainer.py` if needed |
| `Model file not found` | Model was not uploaded. Check `models/phishing_model/` exists in the Space files |
| `Build timeout` | Usually happens with very large uploads. Use Git-based upload instead |

### App Loads But Shows Fallback Mode

If the app says "Trained model not found locally" even after deployment:

1. Go to your Space's **Files** tab
2. Verify `models/phishing_model/config.json` exists (not just the folder)
3. If the model files weren't uploaded properly, re-upload them
4. HuggingFace Spaces restarts automatically when files change

### App is Very Slow

- First load is slow because the model (~260 MB) must be downloaded from HuggingFace storage into the Space's RAM
- SHAP analysis on CPU is inherently slow (30 seconds per prediction)
- To speed it up: For production use, consider reducing `max_evals=50` to `max_evals=30` in `streamlit_app/app.py`
- The CPU basic tier is free but slow. CPU upgraded tiers are faster ($0/month for upgrade)

### Space Goes to Sleep

Free HuggingFace Spaces go to sleep after 48 hours of inactivity. When someone visits after sleeping:
- The app takes 30-60 seconds to cold-start (download model, initialize SHAP)
- After that, it runs normally
- This is normal behavior for the free tier

---

## One-Page Quick Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GOOGLE COLAB (15 min)                                            │
│     ├── https://colab.research.google.com                            │
│     ├── File → Upload → training_notebook.ipynb                      │
│     ├── Runtime → Change runtime → T4 GPU                            │
│     ├── Runtime → Run all                                            │
│     ├── Wait 10-15 min for training                                  │
│     └── Download phishing_model.zip + tokenizer.zip                  │
│                                                                      │
│  2. UNZIP MODEL (2 min)                                              │
│     ├── Unzip into models/phishing_model/                            │
│     └── Unzip into models/tokenizer/                                 │
│                                                                      │
│  3. HUGGINGFACE SPACES (5 min)                                       │
│     ├── Go to https://huggingface.co/new-space                       │
│     ├── Name: genai-phishing-detector                                │
│     ├── SDK: Streamlit                                               │
│     ├── Hardware: CPU basic (Free)                                   │
│     ├── Upload all files + model                                     │
│     └── Wait 3-5 min for build                                       │
│                                                                      │
│  4. DONE 🎉                                                          │
│     └── https://YOUR_USERNAME-genai-phishing-detector.hf.space       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Total time: ~25 minutes | Total cost: $0**
