# Bearst Cancer Prediction API

This is a production-ready **FastAPI** service that predicts breast cancer (benign or malignant) using a trained Logistic Regression model along with a StandardScaler.
The API loads both the model and scaler. You send it 30 numerical features, and it gives you a prediction:

- '0' → Benign
- '1' → Malignant

---

## 🚀 Features

- FastAPI backend with auto-generated Swagger docs
- Real-time prediction at (`/predict`)
- Uses a pre-trained Logistic Regression model
- Features are scaled with StandardScaler
- Project is Clean and modular
- Ready to deploy anywhere

---

##  📁 Project Structure (API Section)

``` bash
src/
│
├── api.py           # FastAPI app
├── predict.py       # Loads model, handles predictions
├── model.pkl        # Trained model file
├── scaler.pkl       # StandardScaler file
└── init.py
```

``` code
> Note: this README is just about the API.
> For details on training the model, check the main `README.md`.
```

---

## ▶️ How to Run the API locally

### 1. Create and activate environment

``` bash
conda create -n Tensor_GPU python=3.9
conda activate Tensor_GPU
```

### 2. Install dependecies

``` bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

``` bash
python -m uvicorn src.api:app --reload
```

### 4. Open Swagger UI

``` code
http://127.0.0.1:8000/docs
```

# 📡 API Endpoint

## POST /predict

This endpoint tells you if a tumor is benign or malignant.

## Request Body Example

``` Json
{
    "data": [
        14.5, 20.3, 95.0, 600.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.05, 0.3, 1.2, 2.5, 25.0, 0.005, 0.02, 0.03, 0.01, 0.02, 0.003, 16.0, 25.0, 110.0, 800.0, 0.12, 0.3, 0.4, 0.15, 0.25, 0.07
    ]
}
```

## Response Example

``` Json
{
    "prediction": 1
}
```

# 📝 Notes

. Input must be exactly 30 numbers.
. Output is always an integer (0 or 1).
. The model and scaler load once at server startup so things stay fast.

# 📌 Future Plans

. Build a Streamlit front-end
. Add more ML models
. improve validation and error handling
. Deploy the API online (Render, Railway, HuggingFace Spaces, etc.)

# 📧 Contact
For questions or suggestion,you can open an issue or a pull request.