# 🔬 Breast Cancer Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-brightgreen)

Predicting breast cancer **(Benign vs Malignant)** using classical ML models, deploy as an interactive web app with **Streamlit**.


## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Modeling](#-modeling)
- [How to Run](#️-how-to-run)
- [Key Findings](#-key-findings)
- [Author](#-author)

---


## 📋 Overview

This project focuses on predicting breast cancer (benign vs malignant) using classical machine learning models on the well-known Breast Cancer Wisconsin dataset from `scikit-learn`.

The goal is to:
- Load and explore the dataset
- Build baseline and improved classification models
- Evaluate models performance using standard metrics
- Save the final model and scaler for future use

----

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 numerical features |
| Target | 0 → Malignant, 1 → Benign |
| Missing values | None |


## Class distribution:
https://claude.ai/chat/images/class_distribution.png

**Basic checks performed:**
- `df.head()` inspect first rows
- `df.shape` → (569, 31)
- `df.info()` dtypes and non-null counts
- `df.isnull().sum()` → No missing values
- `df['target'].value_counts()` - class distribution


----
## 🎯 Results
| Model | Accuracy | F1 Score | AUC |
|-------|---------:|---------:|----:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9954 |
| Random Forest | 0.9561 | 0.9655 | 0.9931 |


> **Best model:** Logistic Regression with 98.25% accuracy

## 📁 Project Structure

```bash
Breast_Cancer_Prediction/
|
├── noteBooks/
│ ├── 01_Data_Loading_and_EDA.ipynb
│ ├── 02_Modeling.ipynb
|
├── models/
│ ├── logistic_regression_model.pkl
│ ├── scaler.pkl
|
├── src/
│ ├── predict.py
|
├── README.md
├── requirements.txt
└── .gitignore
```


## 🧠 Modeling
### 1. Train/Test Split

- Features: All columns except the target
- Target: `target` column
- Split: %80 Train / %20 Test
- Stratified: maintains class proportions

```Python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size = 0.2,
                                    random_state = 42,
                                    stratify = y)
```


### 2. Logistic Regression (with StandardScaler)

```Python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter = 2000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Results:
- Accuracy: ~0.956
- Confusion Matrix:

```Code
[[39  3]
 [2  70]]
```

### 3. Random Forest

```Python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 200,
                                  max_depth = None,
                                  random_state = 42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
print("RandomForest Accuracy: ", rf_acc)

print("\nClassification Report: \n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred_rf))
```

### 4. 💾 Saved Artifacts

```Python
import joblib

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

| File | Description |
|------|-------------|
| 'logistic_regression_model.pkl' | Trained Logistic Regression Model |
| 'scaler.pkl' | Fitted StandardScaler |

---


## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Hossein0213/breast_cancer_prediction.git
cd breast_cancer_prediction
```

### 2. Create a virtual environment (Optional)
```bash
python -m venv venv
source venv/bin/activate # Windows: venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
- Open with Jupyter Notebook
- Or VS Code with Jupyter Extension

### 5. Run Streamlit app
```bash
streamlit run streamlit_app.py
```

---

## 💡 Key Findings

- Dataset is ** clean ** - no missing values across all 30 features
- ** Logistic Regression ** outperforms Random Forest (98.2% vs 95.6%)
- 'StandardScaler' is essential - improves logistic Regression significantly
- Most informative featuers: 'mean radius', 'mean concave points', 'worst perimeter'

---

## 👤 Author

**Hossein** — AI & Robotics Engineer
Master's degree in AI and Robotics

[![GitHub](https://img.shields.io/badge/GitHub-Hossein0213-black)](https://github.com/Hossein0213)