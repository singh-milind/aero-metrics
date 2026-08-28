

from fastapi import APIRouter
from src.api.schemas.predictor.predictor import PredictorInput, prepare_input
import joblib
router = APIRouter()

import numpy as np

@router.post("/predict_pm25")
def predict(request: PredictorInput):

    X = prepare_input(request)
    model = joblib.load("src/model_building/predictor/pm25_model/pm25_predictor.pkl")
    expected_features = model.get_booster().feature_names


    X = X[expected_features]


    prediction = model.predict(X)

    return {
        "prediction": prediction.tolist()
    }

@router.post("/predict_pm10")
def predict(request: PredictorInput):
    X = prepare_input(request)
    model_pm10 = joblib.load("src/model_building/predictor/pm10_model/pm10_predictor.pkl")
    model_pm25 = joblib.load("src/model_building/predictor/pm25_model/pm25_predictor.pkl")
    expected_features = model_pm10.get_booster().feature_names

    X = X[expected_features]

    ratio = model_pm10.predict(X)
    pm25_prediction = model_pm25.predict(X)
    pm10_prediction = ratio * pm25_prediction

    return {
        "prediction": pm10_prediction.tolist()
    }