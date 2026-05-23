<div align="center">

# 📊 Smart Data Preprocessing and Model Advisor

### An end-to-end AutoML web app that cleans your data, detects the ML problem type, trains multiple models, visualizes everything, and exports a full report — all without writing a single line of code.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 What is this project?

**Smart Data Preprocessing and Model Advisor** is a web-based AutoML assistant built with Python and Streamlit. Upload any CSV or Excel dataset (or paste a URL from Kaggle, GitHub, or Google Sheets) and the app automatically:

- ✅ Cleans your data — missing values, duplicates, outliers
- ✅ Scores data quality from 0 to 100
- ✅ Detects the ML problem type — Classification, Regression, or Clustering
- ✅ Trains all recommended models and compares them side by side
- ✅ Shows 12+ interactive visualizations
- ✅ Lets you build custom charts with any column combination
- ✅ Generates a downloadable report in TXT, Markdown, or PDF

---

## ✨ Features

### 📂 Data Loading
| Method | Details |
|--------|---------|
| File Upload | CSV (`.csv`) and Excel (`.xlsx`, `.xls`) |
| URL Import | GitHub raw links, Google Sheets, Kaggle public files, any public CSV URL |
| Auto-fix | GitHub `blob` URLs → `raw` URLs converted automatically |
| Auto-fix | Google Sheets share URLs → CSV export URLs converted automatically |

### 🧹 Data Cleaning Pipeline
| Step | What it does |
|------|-------------|
| Fix Data Types | Converts numeric strings stored as text to actual numbers |
| Handle Missing Values | Numeric → Median · Categorical → Mode |
| Remove Duplicates | Drops exact duplicate rows, resets index |
| Cap Outliers (IQR) | Winsorizes values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR |

### ⭐ Data Quality Score (0–100)
- **Missing values** → up to 40 pts
- **Duplicate rows** → up to 30 pts
- **Constant columns** → up to 20 pts
- **Mixed-type columns** → up to 10 pts

### 🤖 Auto Problem Detection + Model Training
| Problem Type | Condition | Models Trained |
|---|---|---|
| Classification | Categorical target or ≤ 10 unique values | Random Forest · Logistic Regression · Decision Tree |
| Regression | Continuous numeric target (> 10 unique) | Linear Regression · Decision Tree · Random Forest |
| Clustering | No target column selected | K-Means (k auto-selected) |

### 📈 Visualizations (12 chart tabs + custom builder)
**Auto-generated tabs:**
- Missing Values Heatmap
- Correlation Heatmap
- Histograms
- Boxplots
- Violin Plots
- KDE Density Plots
- Categorical Bar Charts
- Pie Charts
- Pairplot (scatter matrix)
- Count Plots
- Descriptive Statistics
- Skewness & Kurtosis

**Custom Chart Builder (13 chart types):**
> Scatter · Line · Area · Bar · Histogram · Box · Violin · KDE · Pie · Count · Correlation Heatmap · Pairplot · Pivot Heatmap

### 📋 Report Export
| Format | File | Best for |
|--------|------|---------|
| Plain Text | `data_report.txt` | Lab notebook, quick sharing |
| Markdown | `data_report.md` | GitHub README, VS Code preview |
| PDF | `data_report.pdf` | College submission, presentation |

---

## 🗂️ Project Structure

```
📁 smart-data-advisor/
│
├── 📄 main.py          ← Backend engine (all logic, no Streamlit)
│   ├── Data loading & URL import
│   ├── Data cleaning pipeline
│   ├── Quality scoring
│   ├── Feature engineering
│   ├── Problem type detection
│   ├── Multi-model training & comparison
│   ├── 15+ visualization builders
│   ├── Report card generator
│   └── TXT / MD / PDF export
│
├── 📄 app.py           ← Streamlit frontend (UI only, imports from main.py)
│   ├── Sidebar (upload, URL, target, split slider)
│   ├── Tab 1: Overview
│   ├── Tab 2: Clean Data
│   ├── Tab 3: Analyze & Charts (12 chart tabs)
│   ├── Tab 4: Custom Chart Builder
│   ├── Tab 5: Train & Compare
│   └── Tab 6: Report Card + Export
│
└── 📄 README.md
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/Suresh-DG/Smart-Data-Preprocessing-and-Model-Advisor.git
cd Smart-Data-Preprocessing-and-Model-Advisor
```

### Step 2 — Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn reportlab openpyxl requests
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.30+ | Web app framework / UI |
| `pandas` | 2.0+ | Data loading, cleaning, manipulation |
| `numpy` | 1.24+ | Numerical operations |
| `scikit-learn` | 1.3+ | ML models, preprocessing, metrics |
| `matplotlib` | 3.7+ | Base plotting library |
| `seaborn` | 0.12+ | Statistical visualizations |
| `requests` | 2.31+ | URL dataset fetching |
| `reportlab` | 4.0+ | PDF report generation |
| `openpyxl` | 3.1+ | Excel file reading |

---

## 🚀 How to Use

```
1. Open the app → streamlit run app.py

2. SIDEBAR
   ├── Choose "Upload File" or "Import from URL"
   ├── Select target column (or leave None for clustering)
   └── Adjust train/test split (60% to 90%)

3. OVERVIEW TAB
   └── See dataset shape, missing values, quality score, data preview

4. CLEAN DATA TAB
   ├── Click "Run Cleaning Pipeline"
   ├── See before/after comparison with delta metrics
   └── Download cleaned CSV

5. ANALYZE & CHARTS TAB
   ├── See detected problem type + recommended models
   └── Explore 12 chart types across sub-tabs

6. CUSTOM CHARTS TAB
   ├── Pick any chart type from dropdown
   ├── Select columns and options
   └── Click "Generate Chart"

7. TRAIN & COMPARE TAB
   ├── Click "Train All Models & Compare"
   ├── See comparison table + bar chart
   ├── View confusion matrix (classification)
   └── View feature importances

8. REPORT CARD TAB
   ├── See full summary — quality, correlations, results
   └── Download report as TXT / Markdown / PDF
```

---

## 🔗 URL Import — Supported Sources

| Source | Example |
|--------|---------|
| **GitHub** | `https://github.com/user/repo/raw/main/data.csv` |
| **GitHub (auto-fix)** | `https://github.com/user/repo/blob/main/data.csv` → auto-converted |
| **Google Sheets** | Publish to web → Copy CSV link |
| **Kaggle** | Raw file download link from dataset Files tab |
| **UCI ML Repository** | Direct `.csv` link |
| **data.gov / any public CSV** | Any direct HTTP/HTTPS `.csv` URL |

---

## 🧠 ML Models Used

### Classification
- `RandomForestClassifier` (100 trees, random_state=42)
- `LogisticRegression` (max_iter=1000)
- `DecisionTreeClassifier`
- **Metric:** Accuracy %

### Regression
- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor` (100 trees)
- **Metrics:** RMSE, R² Score

### Clustering
- `KMeans` (k auto-selected = `min(4, n//10 + 2)`)
- **Metric:** Inertia

---

## 📊 Evaluation Metrics Explained

| Metric | Type | Meaning |
|--------|------|---------|
| **Accuracy** | Classification | % of test samples correctly predicted |
| **RMSE** | Regression | Average prediction error (lower = better) |
| **R² Score** | Regression | How much variance the model explains (closer to 1 = better) |
| **Inertia** | Clustering | Sum of squared distances to cluster centres (lower = tighter clusters) |
| **Confusion Matrix** | Classification | Shows correct vs incorrect predictions per class |

---

## 🗺️ Pipeline Flowchart

```
Upload / URL
     │
     ▼
Load Dataset (CSV / Excel)
     │
     ▼
Overview — Shape, Types, Quality Score
     │
     ▼
Clean Data
  ├── Fix Data Types
  ├── Fill Missing Values (Median / Mode)
  ├── Remove Duplicates
  └── Cap Outliers (IQR Winsorization)
     │
     ▼
Analyze — Detect Problem Type
  ├── No target     → Clustering
  ├── Categorical   → Classification
  └── Continuous    → Regression
     │
     ▼
Feature Engineering
  ├── Label Encode (≤ 20 unique values)
  ├── One-Hot Encode (> 20 unique values)
  └── Standard Scale (all numeric features)
     │
     ▼
Train All Models → Compare → Best Model
     │
     ▼
Report Card → Export TXT / MD / PDF
```

---

## 🔮 Future Improvements

- [ ] Prediction on new data using the best trained model
- [ ] Kaggle API integration (browse & download datasets by name)
- [ ] SVM and KNN in model comparison
- [ ] Deploy on Streamlit Cloud
- [ ] Mobile-friendly layout

---

## 🎓 About

This project was built as a part of **2nd Year, Mini Project**:

- Data Preprocessing & EDA
- Supervised Learning (Classification & Regression)
- Unsupervised Learning (Clustering)
- Feature Engineering
- Model Evaluation
- Web App Development with Streamlit

**Tech Stack:** Python · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · ReportLab · Requests

---


<div align="center">

Made with ❤️ by [Suresh DG](https://github.com/Suresh-DG)

⭐ **Star this repo if you found it helpful!** ⭐

</div>
