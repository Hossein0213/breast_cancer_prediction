#Breast Cancer Prediction using Machine Learning

## Overview

This project focuses on predicting breast cancer (benign vs malignant) using classical machine learning models on the well-known Breast Cancer Wisconsin dataset from 'sciki-learn'.

The goal is to:
- Load and explore the dataset
- Build baseline and improved classification models
- Evaluate models performance using standard metrics
- Save the final model and scaler for future use

----

# Dataset
- Source: 'sklearn.datasets.load_breast_cancer'
- Samples: 569
features: 30 numerical features
- Target:
    - '0' → Malignant
    - '1' → Benign


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
├── NoteBooks/
│ ├── 01_Data_Loading.ipynb
│ ├── breast_cancer_logistic_regression_model.pkl
│ ├── scaler.pkl
├── README.md
├── .gitignore
└── (other project files)

```Python
from sklearn.model_selection import train_test_split