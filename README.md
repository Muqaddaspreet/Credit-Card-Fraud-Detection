# Credit Card Fraud Detection

> **Author:** Muqaddaspreet Singh  
> **Domain:** Machine Learning · Fraud Analytics · Imbalanced Learning

## 🔖 Tags
`python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `imbalanced-learn` · `fraud-detection` · `classification` · `roc-auc`

---

## 📌 Overview

This repository explores **credit card fraud detection** using classical machine-learning classifiers on a highly **imbalanced dataset**. The workflow covers **EDA**, **data preprocessing**, **class imbalance handling** (Random Under-Sampling, Random Over-Sampling, and **SMOTE**), model training, cross-validation, and **ROC-AUC–based** evaluation with confusion matrices and precision/recall/F1 reports. :contentReference[oaicite:0]{index=0}

---

## 🧠 Problem Statement

Classify each transaction as **Fraud** or **Legit** while dealing with extreme **class imbalance** (≈ 0.17% fraud vs 99.83% legit) and preserving generalization. :contentReference[oaicite:1]{index=1}

---

## 🗂️ Dataset

- **Features:** 31 columns (anonymized components + `Time`, `Amount`, and `Class`).  
- **Imbalance:** Legit ≈ **99.82%**, Fraud ≈ **0.17%** (severe skew).  
- **Integrity:** **No missing values** detected.  
- **Scaling:** `Time` and `Amount` standardized (StandardScaler) before modeling. :contentReference[oaicite:2]{index=2}

> 📌 For some experiments, a **balanced subset** was created via **Random Under-Sampling**: **984** rows total → **492 Fraud** + **492 Legit**. :contentReference[oaicite:3]{index=3}

---

## 🏗️ Approach & Pipeline

1. **EDA & Integrity Checks** – schema, nulls, class distribution. :contentReference[oaicite:4]{index=4}  
2. **Preprocessing** – standardize `Time`, `Amount`. :contentReference[oaicite:5]{index=5}  
3. **Imbalance Strategies**  
   - Random **Under-Sampling** (RUS).  
   - Random **Over-Sampling** (ROS).  
   - **SMOTE** (synthetic minority over-sampling). :contentReference[oaicite:6]{index=6}  
4. **Modeling** – Logistic Regression, KNN, SVM, Gaussian Naïve Bayes, Decision Tree. :contentReference[oaicite:7]{index=7}  
5. **Evaluation** – Confusion Matrix, Precision/Recall/F1, **ROC-AUC**, ROC curves; **k-fold cross-validation** to sanity-check variance. :contentReference[oaicite:8]{index=8}

---

## 🤖 Models & Key Results (highlights)

- **Logistic Regression**  
  - Strong separability with **ROC-AUC ≈ 97.28%** on the reported experiment.  
  - Detailed confusion matrix and PRF scores included. :contentReference[oaicite:9]{index=9}

- **K-Nearest Neighbors (KNN)**  
  - **ROC-AUC ≈ 93.69%** (sensitive to feature scaling and `k`). :contentReference[oaicite:10]{index=10}

- **SVM · Gaussian Naïve Bayes · Decision Tree**  
  - Trained and evaluated with the same protocol; confusion matrices, PRF reports, ROC curves, and CV scores recorded per balancing strategy. :contentReference[oaicite:11]{index=11}

> ✅ All models were assessed **per balancing technique** (RUS/ROS/SMOTE) to compare trade-offs between precision and recall under skewed distributions. :contentReference[oaicite:12]{index=12}

---

## 📈 Evaluation

- **Confusion Matrix** for error analysis (TP/FP/FN/TN).
- **Precision / Recall / F1-Score** with emphasis on **Recall** for Fraud class.
- **ROC-AUC** and **ROC curves** across models and sampling strategies.
- **k-Fold Cross-Validation** for stability (variance across folds).

---

## 📬 Results & Findings

- Severe class imbalance demands **balancing**; **SMOTE/ROS** improve minority recall, while **RUS** reduces data size to mitigate bias.
- **Logistic Regression** achieved **ROC-AUC ≈ 97.28%** in the reported setting; **KNN ≈ 93.69%**; others show varying trade-offs (see reports).
- Visual diagnostics (**ROC curves** and **confusion matrices**) guide operating-point selection for production thresholds.

---

## ⚖️ Limitations & Future Work

- **Cost-sensitive learning** or **focal loss** to prioritize minority class.
- Explore **tree ensembles** (Random Forest, XGBoost, LightGBM) and **calibrated probabilities**.
- Use **time-aware splits** and **drift detection** for real payment streams.
- Add **threshold tuning** (PR-AUC maximization) and **post-hoc calibration** (Platt/Isotonic).
- Consider **anomaly detection** baselines (Isolation Forest, One-Class SVM).

---

## 📚 References

- Project analysis, methodology, balancing strategies, and model results are adapted from the **“Report on Credit Card Fraud Detection” (PDF)** included with this repository.

---



