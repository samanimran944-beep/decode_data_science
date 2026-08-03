# Project 2 — Supervised Learning: Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A leak-free classification pipeline that detects fraudulent transactions
in a severely imbalanced dataset (< 1% fraud), built around SMOTE
resampling, stratified cross-validation, and threshold-aware evaluation
metrics — never raw accuracy.

## 📌 Problem Statement

In enterprise payment infrastructure, a model that classifies **every**
transaction as legitimate achieves near-perfect accuracy (>99%) while
catching **zero fraud** — a catastrophic outcome. This project builds a
pipeline that is explicitly immune to two traps:

- **Trap #1 — The Illusion of Accuracy**: solved by evaluating on
  Precision, Recall, and ROC-AUC instead of accuracy.
- **Trap #2 — The Data Leakage Catastrophe**: solved by splitting the
  data *first*, then applying SMOTE and scaling **only** to the training
  fold — the test set (and every cross-validation fold's held-out
  partition) never sees synthetic or leaked information.

## 🎯 What This Notebook Does

1. Quantifies the class imbalance and demonstrates why accuracy is a
   misleading metric on this problem.
2. Performs a **stratified train/test split before any resampling**.
3. Implements **SMOTE from first principles** (the exact interpolation
   formula `x_new = x_i + λ(x_nn − x_i)`) using `sklearn.NearestNeighbors`,
   alongside the production-equivalent `imbalanced-learn` snippet.
4. Runs a **leak-free 5-fold stratified cross-validation**, applying
   SMOTE inside every training fold only.
5. Trains and compares **Logistic Regression** and **Random Forest**
   classifiers.
6. Evaluates the final models on an **untouched test set** using
   confusion matrices, ROC curves, Precision-Recall curves, and a full
   classification report.

## 📂 Repository Structure

```
project-2-fraud-detection-pipeline/
├── notebook.ipynb          # Full, executed, end-to-end walkthrough
├── data/
│   └── transactions.csv    # Synthetic imbalanced transaction dataset
├── requirements.txt
└── README.md
```

## 📊 Dataset

`data/transactions.csv` is a **synthetically generated** dataset (20,000
transactions) built to mirror the structure and extreme imbalance of the
well-known credit-card-fraud problem: anonymized PCA-style features
`V1`–`V10`, `Time`, `Amount`, and a binary `Class` target (0.75% fraud
rate). It ships in this repo instead of the real 284,807-row / 150MB+
Kaggle dataset to keep the portfolio repository lightweight.

**Want to run this against the real dataset?** Download
["Credit Card Fraud Detection"](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
from Kaggle, drop `creditcard.csv` into `data/`, and update the
`pd.read_csv(...)` path in Section 1 of the notebook — the rest of the
pipeline works unchanged (with `V1`–`V28` instead of `V1`–`V10`).

## ✅ Results (on the untouched, naturally imbalanced test set)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.566 | 1.000 | 0.723 | 0.9999 |
| **Random Forest** | **0.833** | **1.000** | **0.909** | **0.9999** |

Both models catch **100% of fraud** in the test set; Random Forest does
so with far fewer false positives, making it the stronger candidate for
deployment where false declines carry a customer-experience cost.

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/decodelabs_data_science.git
cd decodelabs_data_science/project-2-fraud-detection-pipeline
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` (SMOTE) ·
`Matplotlib` · `Seaborn`

## 🔑 Key Skills Demonstrated

Handling severe class imbalance, SMOTE mechanics, leak-free
train/test/CV design, classification algorithm comparison, threshold-aware
evaluation (Precision/Recall/ROC-AUC over accuracy), confusion matrix and
ROC/PR curve interpretation.

---
Part of the **DecodeLabs Data Science Industrial Training Kit — 2026 Batch**.
