# Project 1 — Advanced EDA & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Turning a raw, chaotic retail-customer dataset into mathematically clean,
model-ready fuel for machine learning — through statistically justified
missing-data handling, IQR-based outlier control, and engineered features.

## 📌 Problem Statement

Machine learning estimators have zero qualitative reasoning — they are
numerical optimizers operating on coordinate spaces. If low-fidelity,
unrefined data enters the system, the algorithm will flawlessly optimize
for the *wrong* patterns. This project treats data preprocessing as
structural engineering, not janitorial work.

## 🎯 What This Notebook Does

1. **Diagnoses missingness** and applies the correct strategy per column
   based on *how much* is missing — not a one-size-fits-all fix:
   | Missing % | Strategy |
   |---|---|
   | < 5% | Row deletion |
   | 5% – 20% | Median (numeric) / group-wise mode (categorical) imputation |
   | > 20% | K-Nearest-Neighbors imputation |
2. **Detects and neutralizes outliers** with the IQR method, winsorizing
   (clipping) instead of deleting rows to preserve data volume.
3. **Engineers 5 new features** (exceeding the 3-feature requirement):
   `purchase_frequency_score`, `customer_tenure_years`,
   `income_to_spend_ratio`, `annual_spend`, `is_high_value_customer`.
4. **Encodes categorical variables correctly** — ordinal encoding for the
   naturally ordered `membership_tier`, one-hot encoding for nominal
   fields (`gender`, `region`, `acquisition_channel`).
5. **Checks for multicollinearity** via a Pearson correlation matrix
   before declaring the dataset ML-ready.

## 📂 Repository Structure

```
project-1-advanced-eda-feature-engineering/
├── notebook.ipynb          # Full, executed, end-to-end walkthrough
├── data/
│   ├── retail_customers_raw.csv     # Raw, messy input (6,000 rows)
│   └── processed_customers.csv      # Clean, encoded, model-ready output
├── requirements.txt
└── README.md
```

## 📊 Dataset

`data/retail_customers_raw.csv` is a **synthetically generated** dataset
(6,000 customers × 12 columns) built to deliberately mirror real-world
data-quality problems: three different missingness regimes, injected
outliers in `age`, `annual_income`, and `avg_purchase_amount`, a skewed
income distribution, and a mix of nominal/ordinal categorical fields.

## ✅ Results

- Started with **6,000 rows / 12 columns** of raw, messy data.
- After the `<5%`-missing row-deletion pass, statistical imputation,
  and KNN imputation, the pipeline ends with **0 remaining missing
  values**.
- Final processed dataset: **5,587 rows × 28 columns** after feature
  engineering and one-hot encoding — fully model-ready.

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/decodelabs_data_science.git
cd decodelabs_data_science/project-1-advanced-eda-feature-engineering
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` (`KNNImputer`) ·
`Matplotlib` · `Seaborn`

## 🔑 Key Skills Demonstrated

Statistical imputation strategy selection, IQR-based outlier
winsorization, domain-driven feature engineering, ordinal vs. one-hot
encoding decisions, multicollinearity diagnosis.

---
Part of the **DecodeLabs Data Science Industrial Training Kit — 2026 Batch**.
