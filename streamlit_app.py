import streamlit as st
import requests

st.title("Breast Cancer Prediction App")
st.write("""Enter the 30 numerical features for prediction.""")

# create input fields for 30 features
features = []
for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict"):
    payload = {"data": features}
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)


    if response.status_code == 200:
        results = response.json()["prediction"]

        if results == 1:
            st.success("Prediction: The tumor is predicted to be malignant (1).")
        else:
            st.success("Prediction: The tumor is predicted to be benign (0).")
    
    else:
        st.error(f"Error: {response.status_code} - {response.text}")