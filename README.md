# 🏥 AI-Driven Hospital Readmission Risk Prediction System

An **end-to-end Artificial Intelligence project** that predicts **30-day hospital readmission risk** and presents **actionable, explainable insights** through an interactive dashboard for healthcare decision support.

This system is designed as a **risk-screening and prioritization tool**, not a diagnostic system.

---

## 📌 Project Overview

Hospital readmissions increase healthcare costs, strain hospital resources, and negatively impact patient outcomes.
This project builds an AI-based system that helps healthcare teams:

* Identify patients at **high risk of readmission**
* Prioritize follow-up and care-coordination efforts
* Support **data-driven clinical operations**

The solution emphasizes **interpretability, recall, and real-world usability**.

---

## 🎯 Business Problem

Hospitals need to proactively identify patients likely to be readmitted within 30 days, but:

* Patient data is noisy and highly imbalanced
* Rule-based systems miss complex risk patterns
* Clinical teams have limited resources

A predictive, explainable risk-stratification system is required to support effective care planning.

---

## ✅ Solution Highlights

* 📊 Predicts **patient-level readmission risk probabilities**
* 🚦 Stratifies patients into **Low → Very High risk categories**
* 🧠 Provides **SHAP-based explainability** at patient level
* 🖥️ Interactive **Streamlit dashboard** for analysts and care teams
* 🔁 Complete ML pipeline: preprocessing → modeling → evaluation → visualization
* 📄 Model metadata for traceability and governance

---

## 🧱 System Architecture

```
Raw Hospital Data
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
XGBoost Model Training
        ↓
Risk Stratification (Quantile-based)
        ↓
Explainability (SHAP)
        ↓
Streamlit Dashboard
```

---

## 📂 Project Structure

```
AI-Hospital-Readmission-Risk/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── xgb_classifier_readmission_model.pkl
│   └── model_metadata.json
│
├── reports/
│   └── executive_summary.md
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## 📊 Data Description

* **Dataset**: UCI Diabetes 130-US Hospitals Dataset
* **Size**: ~100,000 hospital encounters
* **Target**: Readmission within 30 days
* **Key Challenges**:

  * Highly imbalanced target variable
  * High-cardinality clinical codes
  * Missing and encoded values (`?`)

---

## ⚙️ Machine Learning Approach

* **Model**: XGBoost Classifier
* **Why XGBoost?**

  * Strong performance on tabular healthcare data
  * Handles non-linear relationships
  * Widely used in industry

### Key Design Decisions

* Optimized for **recall**, not accuracy
* Risk-based thresholding
* Feature selection focused on interpretability
* High-cardinality diagnosis & medication codes intentionally excluded

---

## 📈 Model Evaluation (Risk-Optimized)

```
ROC-AUC Score: ~0.68
High-Risk Recall: ~99%
Accuracy: ~0.15 (intentionally deprioritized)
```

### Interpretation

* High recall ensures **very few high-risk patients are missed**
* Lower accuracy is expected due to class imbalance and aggressive screening
* This aligns with **real healthcare screening use cases**

> ⚠️ In healthcare, missing a high-risk patient is costlier than false positives.

---

## 🚦 Risk Stratification

Patients are grouped using **quantile-based risk bands**:

| Risk Category     | Meaning                |
| ----------------- | ---------------------- |
| 🟢 Low Risk       | Routine follow-up      |
| 🟡 Medium Risk    | Monitor                |
| 🟠 High Risk      | Priority follow-up     |
| 🔴 Very High Risk | Immediate intervention |

This ensures balanced, actionable categories even when probabilities are skewed.

---

## 🧠 Explainability (SHAP)

* Patient-level SHAP explanations
* Identifies top features contributing to readmission risk
* Builds trust and transparency for healthcare stakeholders

---

## 🖥️ Dashboard Features

* 📊 Readmission risk distribution
* 📋 Ranked patient-level risk tables
* 🚨 High & very-high-risk patient identification
* 🧠 SHAP-based explanations per patient
* 🎨 Color-coded risk levels for intuitive understanding

---

## ▶️ How to Run the Project

## Clone repository:

```
git clone https://github.com/girishshenoy16/AI-Hospital-Readmission-Risk.git
cd AI-Hospital-Readmission-Risk
```


### 1️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ Run data pipeline

```bash
python src/preprocessing.py
python src/features.py
python src/train.py
python src/evaluate.py
```

### 4️⃣ Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🧪 Outputs Generated

* Cleaned & engineered datasets
* Trained ML model
* Model metadata (parameters, ROC-AUC, threshold)
* Evaluation metrics
* Interactive dashboard
* Executive summary report

---

## 🚀 Future Improvements

* Integration with EHR systems
* NLP on clinical notes
* Real-time prediction APIs
* Cost-sensitive optimization
* Model monitoring & retraining pipelines

---

## 👤 Author

**Girish Shenoy**
Computer Science Student | Aspiring AI & Business Analyst

This project was built as an **industry-oriented portfolio project**, emphasizing real-world healthcare analytics, explainability, and execution quality.

---

## ⭐ Final Note

This project demonstrates how **applied AI** can support healthcare decision-making by combining predictive modeling, interpretability, and practical visualization — aligned with real hospital workflows.
