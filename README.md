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

### Stage 1 — Model Performance

![Stage 1 Confusion Matrix and ROC](images/01_stage1_confusion_roc.png)

### What Drives Response? (SHAP — Stage 1)

![SHAP Bar Stage 1](images/02_shap_bar_stage1.png)

![SHAP Beeswarm Stage 1](images/03_shap_beeswarm_stage1.png)

**Top drivers of response:**
- Higher **income** customers respond significantly more
- **Wine and meat spend** are the strongest positive signals
- Customers with **teenagers at home** are less likely to respond
- **Recency** matters — recently active customers convert better
- Having a **partner** slightly increases response likelihood

---

## Customer Profiling

### Response Rate by Demographics

![Response by Partner Status](images/04_response_by_partner.png)

![Wine Spend by Response](images/05_wine_spend_by_response.png)

![Feature Analysis Grid](images/06_feature_analysis_grid.png)

---

## The Goldilocks Segmentation

![Segment Distribution](images/09_goldilocks_segmentation.png)

### What Differentiates the Segments?

![ANOVA Top Features](images/08_anova_top_features.png)

---

## Stage 2 — Persuadables vs Sure Things

![Stage 2 Confusion Matrix and ROC](images/10_stage2_confusion_roc.png)

### What Makes a Persuadable? (SHAP — Stage 2)

![SHAP Bar Stage 2](images/11_shap_bar_stage2.png)

![SHAP Beeswarm Stage 2](images/12_shap_beeswarm_stage2.png)

### Age Comparison: Persuadables vs Sure Things

![Age Comparison](images/13_age_comparison_segments.png)

---

## Business Impact — KPI Dashboard

![KPI Dashboard](images/14_kpi_dashboard.png)

By targeting only Persuadables the model **reduces campaign cost** while maintaining conversion volume — avoiding both wasted spend on Lost Causes and unnecessary contacts with Sure Things who would convert regardless.

---

## Methodology

- **Imbalance handling**: `class_weight='balanced'` and `scale_pos_weight` across all models
- **Hyperparameter tuning**: Optuna with 50 trials per model, optimizing PR-AUC
- **Models compared**: LightGBM ✅, XGBoost, CatBoost, Logistic Regression, SVM, MLP
- **Explainability**: SHAP values for both stages — bar plots, beeswarm, and dependence plots
- **Threshold optimization**: precision-recall curve used to find the optimal classification cutoff
- **Feature engineering**: customer age from birth year, tenure in days from join date, partner status as binary

---

## Dataset

[UCI Marketing Campaign Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) — customer personality and purchase behavior data from a retail company.

---

## Stack

`Python` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `SHAP` · `Scikit-learn` · `Pandas` · `Matplotlib` · `Seaborn` · `SciPy`

---

## How to Run

```bash
git clone https://github.com/yourusername/customer-response-prediction
cd customer-response-prediction
pip install -r requirements.txt
jupyter notebook notebook/01_Study.ipynb
```
