import joblib
import numpy as np

model = joblib.load("models/breast_cancer_logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return int(prediction[0])