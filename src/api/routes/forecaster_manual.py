

from fastapi import APIRouter
from src.api.schemas.forecaster.manual.forecaster import  ForecasterInput, prepare_input_pm25, prepare_input_pm10
import pandas as pd
import joblib
router = APIRouter()
import numpy as np
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")
router = APIRouter()

pm25_t_model = joblib.load("/models/forecaster/pm25/t_model/pm25_forecaster_t.pkl")
pm25_t12_model = joblib.load("/models/forecaster/pm25/t12_model/pm25_forecaster_t12.pkl")
pm25_t24_model = joblib.load("/models/forecaster/pm25/t24_model/pm25_forecaster_t24.pkl")
pm25_t48_model = joblib.load( "/models/forecaster/pm25/t48_model/pm25_forecaster_t48.pkl")

pm10_t_model = joblib.load("/models/forecaster/pm10/t_model/pm10_forecaster_t.pkl")
pm10_t12_model = joblib.load("/models/forecaster/pm10/t12_model/pm10_forecaster_t12.pkl")
pm10_t24_model = joblib.load("/models/forecaster/pm10/t24_model/pm10_forecaster_t24.pkl")
pm10_t48_model = joblib.load("/models/forecaster/pm10/t48_model/pm10_forecaster_t48.pkl")

expected_features_pm25 = pm25_t_model.get_booster().feature_names
expected_features_pm10_t = pm10_t_model.get_booster().feature_names
expected_features_pm10_rest = pm10_t12_model.get_booster().feature_names

@router.post("/forecast_manual")
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
    
    pm10_t_prediction = ratio_t_prediction * pm25_t_prediction
    pm10_t12_prediction = ratio_t12_prediction * pm25_t12_prediction
    pm10_t24_prediction = ratio_t24_prediction * pm25_t24_prediction
    pm10_t48_prediction = ratio_t48_prediction * pm25_t48_prediction

    return {
    "target_time": start_time.isoformat(),
        "pm25_predictions": {
        "t": round(float(pm25_t_prediction[0]), 4),
        "t12": round(float(pm25_t12_prediction[0]), 4),
        "t24": round(float(pm25_t24_prediction[0]), 4),
        "t48": round(float(pm25_t48_prediction[0]), 4)
    },
    "pm10_predictions": {
        "t": round(float(pm10_t_prediction[0]), 4),
        "t12": round(float(pm10_t12_prediction[0]), 4),
        "t24": round(float(pm10_t24_prediction[0]), 4),
        "t48": round(float(pm10_t48_prediction[0]), 4)
    }
    }