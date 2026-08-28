

from fastapi import APIRouter
from urllib3 import request
from src.api.schemas.forecaster.auto.forecaster import  ForecasterInput, prepare_input_pm25, prepare_input_pm10
import joblib
router = APIRouter()
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")
router = APIRouter()

pm25_t_model = joblib.load("src/model_building/forecaster/pm25_model/t_model/pm25_forecaster_t.pkl")
pm25_t12_model = joblib.load("src/model_building/forecaster/pm25_model/t12_model/pm25_forecaster_t12.pkl")
pm25_t24_model = joblib.load("src/model_building/forecaster/pm25_model/t24_model/pm25_forecaster_t24.pkl")
pm25_t48_model = joblib.load( "src/model_building/forecaster/pm25_model/t48_model/pm25_forecaster_t48.pkl")

pm10_t_model = joblib.load("src/model_building/forecaster/pm10_model/t_model/pm10_forecaster_t.pkl")
pm10_t12_model = joblib.load("src/model_building/forecaster/pm10_model/t12_model/pm10_forecaster_t12.pkl")
pm10_t24_model = joblib.load("src/model_building/forecaster/pm10_model/t24_model/pm10_forecaster_t24.pkl")
pm10_t48_model = joblib.load("src/model_building/forecaster/pm10_model/t48_model/pm10_forecaster_t48.pkl")

expected_features_pm25 = pm25_t_model.get_booster().feature_names
expected_features_pm10_t = pm10_t_model.get_booster().feature_names
expected_features_pm10_rest = pm10_t12_model.get_booster().feature_names


@router.post("/forecast_pm25_auto")
def predict(request: ForecasterInput):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")
    X = prepare_input_pm25(request, target_time=start_time, now_time=now_time)
    X12 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    X24 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    X48 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    X = X[expected_features_pm25]
    X12 = X12[expected_features_pm25]
    X24 = X24[expected_features_pm25]
    X48 = X48[expected_features_pm25]

    t_prediction = pm25_t_model.predict(X)
    t12_prediction = pm25_t12_model.predict(X12)
    t24_prediction = pm25_t24_model.predict(X24)
    t48_prediction = pm25_t48_model.predict(X48)

    return {
    "target_time": start_time.isoformat(),
    "predictions": {
        "t": round(float(t_prediction[0]), 4),
        "t12": round(float(t12_prediction[0]), 4),
        "t24": round(float(t24_prediction[0]), 4),
        "t48": round(float(t48_prediction[0]), 4)
    }
}

@router.post("/forecast_pm10_auto")
def predict(request: ForecasterInput):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")

    pm25_X = prepare_input_pm25(request, target_time=start_time, now_time=now_time)
    pm25_X12 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm25_X24 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm25_X48 = prepare_input_pm25(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    pm25_X = pm25_X[expected_features_pm25]
    pm25_X12 = pm25_X12[expected_features_pm25]
    pm25_X24 = pm25_X24[expected_features_pm25]
    pm25_X48 = pm25_X48[expected_features_pm25]
    
    pm10_X = prepare_input_pm10(request, target_time=start_time, now_time=now_time)
    pm10_X12 = prepare_input_pm10(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm10_X24 = prepare_input_pm10(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm10_X48 = prepare_input_pm10(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    pm10_X = pm10_X[expected_features_pm10_t]
    pm10_X12 = pm10_X12[expected_features_pm10_rest]
    pm10_X24 = pm10_X24[expected_features_pm10_rest]
    pm10_X48 = pm10_X48[expected_features_pm10_rest]

    pm25_t_prediction = pm25_t_model.predict(pm25_X)
    pm25_t12_prediction = pm25_t12_model.predict(pm25_X12)
    pm25_t24_prediction = pm25_t24_model.predict(pm25_X24)
    pm25_t48_prediction = pm25_t48_model.predict(pm25_X48)
    
    ratio_t_prediction = pm10_t_model.predict(pm10_X)
    ratio_t12_prediction = pm10_t12_model.predict(pm10_X12)
    ratio_t24_prediction = pm10_t24_model.predict(pm10_X24)
    ratio_t48_prediction = pm10_t48_model.predict(pm10_X48)
    
    t_prediction = ratio_t_prediction * pm25_t_prediction
    t12_prediction = ratio_t12_prediction * pm25_t12_prediction
    t24_prediction = ratio_t24_prediction * pm25_t24_prediction
    t48_prediction = ratio_t48_prediction * pm25_t48_prediction

    return {
    "target_time": start_time.isoformat(),
    "predictions": {
        "t": round(float(t_prediction[0]), 4),
        "t12": round(float(t12_prediction[0]), 4),
        "t24": round(float(t24_prediction[0]), 4),
        "t48": round(float(t48_prediction[0]), 4)
    }
    }