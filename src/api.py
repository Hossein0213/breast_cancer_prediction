from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from .predict import predict

app = FastAPI(title="Bearst Cancer Prediction API", version="0.1.0")

class Features(BaseModel):
    data: list[float]

    @validator("data")
    def validate_length(cls, v):
        if len(v) != 30:
            raise ValueError("Input must contain exactly 30 numerical features.")
        return v
    
@app.post("/predict", summary="Make Prediction", description="Predicts whether a tumor is benign (0) or malignant (1).")
def make_prediction(features: Features):
    try:
        result = predict(features.data)
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail= str(e))