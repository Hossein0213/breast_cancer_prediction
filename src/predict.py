import joblib
import numpy as np

model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]