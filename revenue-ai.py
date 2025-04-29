# file: restaurant_revenue_ai.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
import os

app = FastAPI()

# Load or initialize the revenue model
revenue_model_path = "revenue_model.joblib"

if os.path.exists(revenue_model_path):
    revenue_model = joblib.load(revenue_model_path)
else:
    revenue_model = SGDRegressor()
    X_init = np.array([[0]])  # Dummy data to initialize
    y_init = np.array([0])
    revenue_model.partial_fit(X_init, y_init)

# Request models
class RevenueTrainItem(BaseModel):
    num_people: int
    expected_revenue: float

class CrowdInput(BaseModel):
    num_people: int

@app.post("/train_revenue")
def train_revenue(data: List[RevenueTrainItem]):
    X = np.array([[item.num_people] for item in data])
    y = np.array([item.expected_revenue for item in data])

    revenue_model.partial_fit(X, y)
    joblib.dump(revenue_model, revenue_model_path)
    return {"message": "Revenue model trained successfully", "trained_samples": len(data)}

@app.post("/predict_revenue")
def predict_revenue(input_data: CrowdInput):
    X_new = np.array([[input_data.num_people]])
    predicted_revenue = revenue_model.predict(X_new)[0]
    return {
        "expected_revenue": round(predicted_revenue, 2)
    }
