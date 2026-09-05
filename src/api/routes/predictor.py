

from fastapi import APIRouter
from src.api.schemas.predictor.predictor import PredictorInput, prepare_input
import joblib
router = APIRouter()

import numpy as np

model_pm10 = joblib.load("/models/predictor/pm10/pm10_predictor.pkl")
model_pm25 = joblib.load("/models/predictor/pm25/pm25_predictor.pkl")

expected_features_pm10 = model_pm10.get_booster().feature_names
expected_features_pm25 = model_pm25.get_booster().feature_names

@router.post("/predict_pm25")
def predict(request: PredictorInput):

    X = prepare_input(request)
    X = X[expected_features_pm25]
    prediction = model_pm25.predict(X)

    return {
        "prediction": prediction.tolist()
    }

@router.post("/predict_pm10")
def predict(request: PredictorInput):
    X = prepare_input(request)
    X = X[expected_features_pm10]
    ratio = model_pm10.predict(X)
    pm25_prediction = model_pm25.predict(X)
    pm10_prediction = ratio * pm25_prediction

    return {
        "prediction": pm10_prediction.tolist()
    }