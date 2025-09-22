from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("Tourist_safety_model.joblib")

app = FastAPI()

class TouristData(BaseModel):
    time_in_red_zone_min: float
    time_near_red_zone_min: float
    red_zone_passes: float
    last_update_gap_min: float
    deviation_km: float

@app.post("/predict")
def predict_safety(data: TouristData):
    # Convert input to numpy array
    X_input = np.array([[
        data.time_in_red_zone_min,
        data.time_near_red_zone_min,
        data.red_zone_passes,
        data.last_update_gap_min,
        data.deviation_km
    ]])
    
    # Predict safety score
    safety_score = model.predict(X_input)[0]
    
    # Determine risk factor
    if safety_score >= 70:
        risk_factor = "High"
    elif 50 < safety_score <= 69:
        risk_factor = "Medium"
    else:
        risk_factor = "Low"
    
    return {
        "safety_score": round(float(safety_score), 2),
        "risk_factor": risk_factor
    }
