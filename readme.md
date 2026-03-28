# 🎯 Customer Response Prediction & Campaign Optimization

> **Can we predict which customers will respond to a marketing campaign — and more importantly, which ones actually need it?**

---

## The Business Problem

A retail company runs direct marketing campaigns and wants to stop wasting budget on the wrong customers. The dataset contains **2,240 customers** with demographic data, purchase history, and whether they responded to a previous campaign (~15% did).

The goal isn't just prediction — it's **actionable segmentation**: identify who to target, who to skip, and who will convert on their own.

---

## What This Project Does

Most marketing ML projects stop at "predict who will respond." This one goes further with a **two-stage modeling approach**:

### Stage 1 — Response Prediction
A LightGBM model trained to identify customers likely to respond to the campaign, optimized for **PR-AUC** to handle the class imbalance (85% non-responders).

### Stage 2 — Persuadables vs Sure Things
Customers are segmented into three groups using their predicted propensity score:

| Segment | Description | Action |
|---|---|---|
| 🔵 **Lost Causes** | Very unlikely to respond | Skip — save the budget |
| 🟠 **Persuadables** | On the fence — campaign can tip them | **Target these** |
| 🟢 **Sure Things** | Will respond regardless | Monitor — don't waste spend |

A second LightGBM model then classifies **Persuadables vs Sure Things** to understand what makes each group unique — the closest proxy to uplift modeling without a control group.

---

## Key Results

- **Stage 1 ROC-AUC**: model successfully separates responders from non-responders well above baseline
- **Campaign ROI improvement**: by targeting only Persuadables, the model reduces contact costs significantly compared to a blind campaign
- **Top drivers of response** (from SHAP):
  - Higher income customers respond more
  - Wine and meat spend are strong positive signals
  - Customers with teenagers at home are less likely to respond
  - Recency matters — recently active customers convert better

- **Persuadables profile** (vs Sure Things):
  - Slightly lower income and spend
  - More likely to have kids/teens at home
  - Less tenure — newer customers who haven't fully committed

---

## Business Impact

| Scenario | Cost | Revenue | ROI |
|---|---|---|---|
| Contact everyone (no model) | High | Moderate | Low |
| Target Persuadables only (model) | Reduced | Maintained | Higher |

The model identifies which ~X% of the customer base deserves campaign spend, cutting wasted contacts while maintaining conversion volume.

---

## Methodology

- **Imbalance handling**: `class_weight='balanced'` and `scale_pos_weight` across all models
- **Hyperparameter tuning**: Optuna with 50 trials per model, optimizing PR-AUC
- **Models compared**: LightGBM, XGBoost, CatBoost, Logistic Regression, SVM, MLP
- **Explainability**: SHAP values for both stages — bar plots, beeswarm, and dependence plots
- **Threshold optimization**: precision-recall curve used to find optimal classification cutoff beyond default 0.5
- **Feature engineering**: customer age from birth year, tenure in days from join date, partner status simplified to binary

---

## Dataset

[UCI Marketing Campaign Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) — customer personality and purchase behavior data from a retail company.

---

## Stack

`Python` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `SHAP` · `Scikit-learn` · `Pandas` · `Matplotlib` · `Seaborn`

---

## How to Run

```bash
git clone https://github.com/yourusername/customer-response-prediction
cd customer-response-prediction
pip install -r requirements.txt
jupyter notebook notebook/01_Study.ipynb
```
