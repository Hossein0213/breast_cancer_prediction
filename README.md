# Breast Cancer Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-brightgreen)

Predicting breast cancer **(Benign vs Malignant)** using classical Ml models, deploy as an intractive web app with **Streamlit**.

## Overview

This project focuses on predicting breast cancer (benign vs malignant) using classical machine learning models on the well-known Breast Cancer Wisconsin dataset from 'sciki-learn'.

The goal is to:
- Load and explore the dataset
- Build baseline and improved classification models
- Evaluate models performance using standard metrics
- Save the final model and scaler for future use

----

# Dataset


| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 numerical features |
| Target | 0 → Malignant, 1 → Benign |
| Missing values | None |


Basic checks performed:
- `df.head()` to inspect the first rows
- `df.shape` → (569, 31)
- `df.info()` to inspect dtypes and non-null counts
- `df.isnull().sum()` → No missing values
- `df['target'].value_counts()` to check class distribution


----
## Project Structure

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


### 🧠 Modeling
## 1. Train/Test Split

- Features: All columns except the target
- Target: Column target
- Split: %80 Train, %20 Test
- Startified: To maintain class proportions

```Python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size = 20,
                                    random_state = 42,
                                    stratify = y)
```


## 2. Logistic Regression (with StandardScaler)

```Python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegrssion(max_iter = 2000)
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

### 🎯 Final Model
Selected Model: Logistic Regression
Preprocessing: StandardScaler

Saved Artifacts:
. breast_cancer_logistic_regression_model.pkl
. scaler.pkl

```Python
import joblib

joblib.dump(model, "breast_cancer_logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

### How to Run
## 1. Clone the repository

```bash
git clone https://github.com/Hossein0213/breast_cancer_prediction.git
cd breast_cancer_prediction
```

## 2. (Optional) Create a virtual environment

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the notebook
* with Jupyter Notbook
* Or VS Code (Jupyter Extension)