# Explainable Diabetes Risk Prediction Dashboard with Clinical Feature Contribution Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb5424.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Batch](https://img.shields.io/badge/Batch%20No-10-informational.svg)](#team-members)

> **Academic Project Submission**  
> **Institution:** Dhanalakshmi College of Engineering  
> **Department:** Department of Artificial Intelligence and Data Science  
> **Batch No:** 10  
> **Submission Date:** 09.10.2026  

---

## 👥 Team Members

| No. | Student Name | Register Number |
|:---:|:---|:---:|
| 1 | **Venkata Sai Pavan G** | `410723243102` |
| 2 | **Revanth Kumar K** | `410723243074` |
| 3 | **Khasim Vali Shaik** | `410713253086` |

---

## 📄 Abstract

Diabetes is a major chronic disease that requires early risk identification and continuous monitoring to reduce the possibility of severe health complications. Conventional diabetes prediction systems often focus mainly on providing a risk classification without clearly explaining how individual clinical factors influence the prediction.

This project presents an **Explainable Diabetes Risk Prediction Dashboard with Clinical Feature Contribution Analysis** to provide both diabetes risk prediction and transparent interpretation of the model’s decision. The system uses patient clinical and demographic attributes such as glucose level, blood pressure, body mass index, age, insulin level, and other relevant health parameters to predict the likelihood of diabetes.

The collected data is preprocessed through data cleaning, biological zero-anomaly missing-value handling, feature transformation, and normalization before being provided to an optimized machine learning classification model. Along with the prediction result, **Explainable AI (XAI)** techniques via **TreeSHAP** are incorporated to analyze the contribution of individual clinical features to the predicted risk. The system visualizes these feature contributions through an interactive dashboard, enabling healthcare professionals and patients to understand which health factors have the greatest influence on the prediction and simulate actionable lifestyle interventions.

---

## 🔬 Mathematical & Explainability Foundation

The core interpretability engine utilizes **Shapley Additive exPlanations (SHAP)** rooted in cooperative game theory. For any patient prediction, the total predicted margin log-odds is decomposed additively into the base population expectation plus individual clinical feature contributions:

$$\ln\left(\frac{P(\text{Diabetes})}{1 - P(\text{Diabetes})}\right) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where:
- $\phi_0$ is the base expected value across the population prior ($~34.9\%$).
- $\phi_i(x)$ is the exact Shapley contribution of clinical feature $i$.
- $\phi_i > 0$ represents a **Risk-Increasing Factor** (rendered in red).
- $\phi_i < 0$ represents a **Protective Factor** (rendered in green).

---

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js** (v18+) & **npm**
- **Python** (v3.10+)

### Option A: Running with npm (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/explainable-diabetes-risk-dashboard.git
cd explainable-diabetes-risk-dashboard

# 2. Setup dependencies (creates virtualenv & installs packages)
npm run setup

# 3. Start the dashboard application
npm start
```

Access the dashboard:
- 🌐 **Interactive Dashboard:** [http://localhost:8000](http://localhost:8000)
- 📚 **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Running with Python directly

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run.py
```

---

## 📋 Available npm Scripts

| Command | Action |
|---|---|
| `npm start` | Launches the live server on `http://localhost:8000` |
| `npm run dev` | Runs the server in development mode |
| `npm test` | Runs the complete unit test and integration API test suite |
| `npm run test:pipeline` | Tests zero-imputation, model training, and SHAP mathematical additivity |
| `npm run test:api` | Tests all FastAPI endpoints (predict, what-if, batch CSV upload) |
| `npm run setup` | Automatically sets up Python virtualenv and installs dependencies |

---

## 🌟 Key Dashboard Features

### 1. Patient Clinical Assessment & Diagnostics
- Evaluates 8 core clinical and demographic biomarkers.
- Immediate 1-click loading of representative clinical archetypes:
  - *Healthy Adult* (Low Risk ~2.6%)
  - *Borderline Pre-Diabetic* (Moderate Risk ~38.1%)
  - *Metabolic Syndrome* (High Risk ~86.9%)
  - *Young Adult with Genetic Risk* (~8.8% with elevated family pedigree)
- Circular animated risk gauge and severity classification (Low / Moderate / High Risk).

### 2. Feature Contribution Analysis (SHAP Waterfall)
- Interactive horizontal diverging bar chart depicting the directional push of each biomarker.
- Comprehensive clinical breakdown table comparing raw values against standard physiological reference ranges (ADA & WHO guidelines).
- Tailored clinical recommendations for every biomarker.

### 3. Interactive "What-If" Intervention Simulator
- Dynamic sliders for modifiable health markers: Glucose, BMI, Blood Pressure, and Serum Insulin.
- Quantifies real-time relative risk reduction (e.g. lowering Glucose to 105 mg/dL and BMI to 24 kg/m² reduces risk by **77.5%**).

### 4. Cohort Batch Screening
- Drag-and-drop CSV screening engine for clinical registries.
- Instant cohort metrics (Low / Moderate / High breakdown) and individual primary risk driver attribution.
- Includes pre-built test cohort: `sample_batch_patients.csv` (25 patients).

### 5. Model Transparency & Validation Metrics
- Evaluated on held-out test split:
  - **Accuracy:** `77.3%`
  - **ROC-AUC Score:** `0.835`
  - **Sensitivity / Recall:** `74.1%`
  - **Specificity:** `79.0%`
  - **F1-Score:** `0.718`
- Interactive Confusion Matrix and Global Feature Importance rankings across the population.

---

## 📊 Clinical Features & Reference Ranges

| Feature | Clinical Parameter | Unit | Physiological Reference Range | Category |
|---|---|:---:|:---:|---|
| **Glucose** | 2-Hour Oral Glucose Tolerance | mg/dL | 70 – 139 | Metabolic |
| **BMI** | Body Mass Index | kg/m² | 18.5 – 24.9 | Anthropometric |
| **BloodPressure** | Resting Diastolic Blood Pressure | mmHg | 60 – 80 | Cardiovascular |
| **Insulin** | 2-Hour Serum Insulin | μU/mL | 16 – 166 | Endocrine |
| **Age** | Chronological Age | years | 18 – 80 | Demographic |
| **DiabetesPedigreeFunction**| Genetic Family History Score | score | 0.08 – 0.45 | Genetics |
| **SkinThickness** | Triceps Skinfold Thickness | mm | 10 – 30 | Anthropometric |
| **Pregnancies** | Gestational History Count | count | 0 – 4 | Reproductive |

---

## 🗂️ Project Directory Structure

```
explainable-diabetes-risk-dashboard/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions automated test workflow
├── backend/
│   ├── __init__.py
│   ├── preprocessor.py            # Biological zero-handling & scaling
│   ├── model.py                   # XGBoost training & metric evaluation
│   ├── xai.py                     # TreeSHAP explainer & What-If simulator
│   ├── api.py                     # FastAPI service & REST endpoints
│   └── artifacts/                 # Serialized model, imputer, and metrics
├── static/
│   ├── index.html                 # Single-page clinical dashboard UI
│   ├── styles.css                 # Accessible healthcare design system
│   └── app.js                     # Chart.js visualization & reactive logic
├── tests/
│   ├── test_pipeline.py           # Unit tests for preprocessing & SHAP
│   └── test_api.py                # Integration tests for REST endpoints
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
├── package.json                   # npm scripts configuration
├── requirements.txt               # Python package dependencies
├── run.py                         # One-click dashboard launcher
├── diabetes.csv                   # Benchmark clinical dataset
└── sample_batch_patients.csv      # Sample cohort for batch screening
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive clinical web dashboard |
| `GET` | `/api/health` | Health check for model and XAI engine |
| `GET` | `/api/metadata` | Returns clinical reference ranges and units |
| `GET` | `/api/sample-patients`| Returns preset clinical archetypes |
| `GET` | `/api/model-stats` | Returns accuracy, ROC-AUC, confusion matrix, and feature importances |
| `POST` | `/api/predict` | Computes diabetes risk and exact Shapley feature attributions |
| `POST` | `/api/what-if` | Simulates intervention changes and calculates delta risk reduction |
| `POST` | `/api/batch-predict` | Uploads CSV for automated multi-patient cohort screening |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Acknowledgments

Developed by **Batch 10** at **Dhanalakshmi College of Engineering**, Department of Artificial Intelligence and Data Science.
