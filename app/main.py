from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="RabTech Real-Time ML Inference API",
    description="Production-style FastAPI service for Iris classification.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    sepal_length_cm: float = Field(..., gt=0)
    sepal_width_cm: float = Field(..., gt=0)
    petal_length_cm: float = Field(..., gt=0)
    petal_width_cm: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: List[float]


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@app.get("/")
def root():
    return {
        "service": "RabTech Real-Time ML Inference API",
        "status": "running",
        "docs": "/docs",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array([[
            request.sepal_length_cm,
            request.sepal_width_cm,
            request.petal_length_cm,
            request.petal_width_cm,
        ]])

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()

        return PredictionResponse(
            predicted_class=CLASS_NAMES[prediction],
            predicted_class_index=prediction,
            probabilities=[round(float(p), 6) for p in probabilities],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
