# Used-Car Price Prediction Using Random Forest Regression

**Domain:** Machine Learning & Predictive Analytics  
**Academic Degree:** Bachelor of Technology in Artificial Intelligence and Data Science  
**Status:** Completed & Fully Verified (Modules 1–7)

---

## 📌 Project Overview
This B.Tech AI & Data Science mini project implements an end-to-end Machine Learning pipeline and an interactive web application to predict the fair market resale value of pre-owned cars in India. 

The system leverages **Random Forest Regression** (200 decision trees) trained on verified historical automotive listing data from CarDekho. The project features rigorous data leakage prevention via scikit-learn `Pipeline` and `ColumnTransformer`, exhaustive 5-fold cross-validation, comprehensive exploratory data analysis, automated 30-point integration testing, and an accessible Streamlit web application.

---

## 🏗️ Project Folder Structure
```text
UsedCarPriceEstimation/
│
├── data/
│   ├── car_data.csv                 # Original raw Indian used-car dataset (301 records)
│   ├── cleaned_car_data.csv         # Cleaned, deduplicated dataset with Car_Age (299 records)
│   └── dataset_info.txt             # Dataset origin, attributes, units, and limitations
│
├── models/
│   └── used_car_price_model.joblib  # Serialized Pipeline (ColumnTransformer + 200-Tree RandomForest)
│
├── outputs/
│   ├── 01_selling_price_distribution.png    # Target distribution histogram with KDE
│   ├── 02_selling_price_boxplot.png         # Outlier inspection boxplot
│   ├── 03_car_age_vs_selling_price.png      # Vehicle depreciation scatter + trend line
│   ├── 04_kms_driven_vs_selling_price.png   # Odometer distance vs. price by fuel type
│   ├── 05_present_price_vs_selling_price.png# Showroom purchase cost vs. resale price
│   ├── 06_categorical_vs_selling_price.png  # Subgroup boxplots (Fuel, Seller, Transmission)
│   ├── 07_correlation_heatmap.png           # Numerical correlation matrix
│   ├── feature_importance.png               # Gini feature importance ranking chart
│   ├── preprocessing_report.txt             # Data cleaning & deduplication audit
│   ├── eda_report.txt                       # Statistical findings & subgroup summaries
│   ├── model_metrics.txt                    # Cross-validation & held-out test evaluation
│   ├── test_predictions.csv                 # 60 test-set predictions alongside actual prices
│   ├── testing_report.txt                   # 30-point automated integration test report
│   ├── viva_questions_answers.md            # 60 viva questions & answers guide
│   ├── code_explanation.md                  # Comprehensive source code walkthrough
│   ├── final_project_report.docx            # Complete academic project report in Word (.docx)
│   └── final_presentation.pptx              # 15-slide academic PowerPoint presentation (.pptx)
│
├── inspect_data.py                  # Module 1: Dataset loading and diagnostics
├── preprocess_data.py               # Module 2: Cleaning, deduplication, Car_Age derivation
├── eda.py                           # Module 3: Visual and statistical EDA
├── train_model.py                   # Module 4: Model training, 5-fold CV, evaluation
├── app.py                           # Module 5: Streamlit web application
├── test_app_logic.py                # Module 5: App inference logic tests (4 vehicle tiers)
├── test_integration.py              # Module 6: 30-point system integration test suite
├── generate_report_docx.py          # Module 7: Academic report generator script
├── generate_presentation_pptx.py    # Module 7: Academic slide deck generator script
└── README.md                        # Project documentation and user guide
```

---

## ⚙️ Technologies & Libraries
- **Language:** Python 3.13 (64-bit)
- **Data Manipulation:** Pandas (v2.2.3), NumPy (v2.2.1)
- **Machine Learning:** Scikit-Learn (v1.9.1) (`ColumnTransformer`, `OneHotEncoder`, `Pipeline`, `RandomForestRegressor`, `cross_validate`)
- **Model Serialization:** Joblib (v1.4.2)
- **Data Visualization:** Matplotlib (v3.10.0), Seaborn (v0.13.2)
- **Web Application:** Streamlit (v1.47.1)
- **Documentation Generation:** python-docx (v1.2.0), python-pptx (v1.0.2)

---

## 🚀 Step-by-Step Execution Guide

All commands are executed using Python 3.13 from the project root in PowerShell:

### 1. Module 1: Dataset Inspection
Inspects raw dataset dimensions, column names, data types, null values, and service history presence:
```powershell
py -3.13 inspect_data.py
```

### 2. Module 2: Data Preprocessing
Removes 2 duplicate rows, derives `Car_Age` relative to reference year 2020, and sets up the leakage-free preprocessor:
```powershell
py -3.13 preprocess_data.py
```

### 3. Module 3: Exploratory Data Analysis (EDA)
Generates 7 statistical figures and writes `outputs/eda_report.txt`:
```powershell
py -3.13 eda.py
```

### 4. Module 4: Model Training & Evaluation
Fits the 200-tree Random Forest pipeline on 80% training data, runs 5-fold cross-validation, evaluates on the 20% test partition, and saves `models/used_car_price_model.joblib`:
```powershell
py -3.13 train_model.py
```

### 5. Module 5: Launch Streamlit Web Application
Starts the local Tornado server and opens the interactive dashboard in your browser:
```powershell
py -3.13 -m streamlit run app.py
```
*(Application runs locally at `http://localhost:8501`)*

### 6. Module 6: System Testing & Integration
Runs unit tests across 4 vehicle tiers and executes the automated 30-point system integration suite:
```powershell
py -3.13 test_app_logic.py
py -3.13 test_integration.py
```

### 7. Module 7: Generate Final Deliverables
Generates the Word project report and PowerPoint slide presentation:
```powershell
py -3.13 generate_report_docx.py
py -3.13 generate_presentation_pptx.py
```

---

## 📊 Model Performance Summary

### A. 5-Fold Cross-Validation (Training Partition — 239 samples)
- **Mean Absolute Error (MAE):** **0.6861 Lakhs** ($\approx$ ₹68,613)
- **Root Mean Squared Error (RMSE):** **1.5026 Lakhs** ($\approx$ ₹1,50,264)
- **$R^2$ Score:** **0.8950 (89.50% variance explained)**

### B. Held-Out Test Set (60 Unseen Samples)
- **Median Absolute Error (MedAE):** **0.4773 Lakhs (₹47,735)** *(50% of test cars are predicted within ₹48k of actual listing price)*
- **Mean Absolute Error (MAE):** **1.4471 Lakhs** (₹1,44,711)
- **Root Mean Squared Error (RMSE):** **3.4495 Lakhs** (₹3,44,954)
- **$R^2$ Score:** **0.5383 (53.83%)**

> **Scientific Note on Test $R^2$:**  
> For 56 of the 60 test vehicles, prediction errors are very small (median error under ₹48,000). However, the random test partition contained 3–4 extreme high-end luxury outliers (e.g., a 14-year-old vehicle originally costing ₹23.73L sold for scrap at ₹2.5L). Because $R^2$ squares residuals, these 4 outliers produce squared penalties exceeding 150 each, heavily dampening test $R^2$ on a compact 60-sample sample.

---

## 🔍 Feature Importance Ranking (Gini / MDI)
1. **`Present_Price` (90.54%):** Original showroom price is the dominant baseline tier determinant.
2. **`Car_Age` (7.13%):** Primary driver of economic and physical depreciation.
3. **`Kms_Driven` (1.33%):** Odometer distance usage proxy.
4. **`Transmission_Manual` (0.57%):** Gearbox convenience preference.
5. **`Seller_Type_Individual` (0.22%):** Dealership vs. direct private seller margin.
6. **`Fuel_Type_Diesel` / `Fuel_Type_Petrol` (0.21%):** Powertrain fuel efficiency.
7. **`Owner` (<0.01%):** Marginal ownership penalty.

---

## ⚠️ Academic Disclaimer & Limitations
1. **Sample Size:** 299 records total. Excellent for budget hatchbacks and mid-tier sedans (₹1L to ₹15L), but exhibits higher variance on rare luxury supercars (>₹25L).
2. **Temporal Window:** Historical dataset compiled up to 2018–2020.
3. **Missing Qualitative Features:** Service logs, accident damage, cosmetic paint scratch rating, and state RTO tax tiers are unrecorded.
