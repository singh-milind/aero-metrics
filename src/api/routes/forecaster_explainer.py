import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.schemas.forecaster.auto.forecaster import (
    ForecasterInput as auto_input,
    prepare_input_pm25 as prepare_input_pm25_auto,
    prepare_input_pm10 as prepare_input_pm10_auto)
from src.api.schemas.forecaster.manual.forecaster import (
    ForecasterInput as manual_input,
    prepare_input_pm25 as prepare_input_pm25_manual,
    prepare_input_pm10 as prepare_input_pm10_manual)
from src.api.schemas.forecaster.reasoning import ForecasterReasoning
from src.api.services.forecaster_reasoning import generate_reasoning
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")
router = APIRouter()

pm25_global_shap_t = joblib.load("src/model_building/forecaster/pm25_model/t_model/pm25_global_shap.pkl")
pm25_global_shap_t12 = joblib.load("src/model_building/forecaster/pm25_model/t12_model/pm25_global_shap.pkl")
pm25_global_shap_t24 = joblib.load("src/model_building/forecaster/pm25_model/t24_model/pm25_global_shap.pkl")
pm25_global_shap_t48 = joblib.load("src/model_building/forecaster/pm25_model/t48_model/pm25_global_shap.pkl")

pm25_explainer_t = joblib.load("src/model_building/forecaster/pm25_model/t_model/pm25_explainer.pkl")
pm25_explainer_t12 = joblib.load("src/model_building/forecaster/pm25_model/t12_model/pm25_explainer.pkl")
pm25_explainer_t24 = joblib.load("src/model_building/forecaster/pm25_model/t24_model/pm25_explainer.pkl")
pm25_explainer_t48 = joblib.load("src/model_building/forecaster/pm25_model/t48_model/pm25_explainer.pkl")

pm25_model_t = joblib.load("src/model_building/forecaster/pm25_model/t_model/pm25_forecaster_t.pkl")
pm25_model_t12 = joblib.load("src/model_building/forecaster/pm25_model/t12_model/pm25_forecaster_t12.pkl")
pm25_model_t24 = joblib.load("src/model_building/forecaster/pm25_model/t24_model/pm25_forecaster_t24.pkl")
pm25_model_t48 = joblib.load("src/model_building/forecaster/pm25_model/t48_model/pm25_forecaster_t48.pkl")

pm10_global_shap_t = joblib.load("src/model_building/forecaster/pm10_model/t_model/pm10_global_shap.pkl")
pm10_global_shap_t12 = joblib.load("src/model_building/forecaster/pm10_model/t12_model/pm10_global_shap.pkl")
pm10_global_shap_t24 = joblib.load("src/model_building/forecaster/pm10_model/t24_model/pm10_global_shap.pkl")
pm10_global_shap_t48 = joblib.load("src/model_building/forecaster/pm10_model/t48_model/pm10_global_shap.pkl")

pm10_explainer_t = joblib.load("src/model_building/forecaster/pm10_model/t_model/pm10_explainer.pkl")
pm10_explainer_t12 = joblib.load("src/model_building/forecaster/pm10_model/t12_model/pm10_explainer.pkl")
pm10_explainer_t24 = joblib.load("src/model_building/forecaster/pm10_model/t24_model/pm10_explainer.pkl")
pm10_explainer_t48 = joblib.load("src/model_building/forecaster/pm10_model/t48_model/pm10_explainer.pkl")

pm10_model_t = joblib.load("src/model_building/forecaster/pm10_model/t_model/pm10_forecaster_t.pkl")
pm10_model_t12 = joblib.load("src/model_building/forecaster/pm10_model/t12_model/pm10_forecaster_t12.pkl")
pm10_model_t24 = joblib.load("src/model_building/forecaster/pm10_model/t24_model/pm10_forecaster_t24.pkl")
pm10_model_t48 = joblib.load("src/model_building/forecaster/pm10_model/t48_model/pm10_forecaster_t48.pkl")


expected_features_pm25_t = pm25_model_t.get_booster().feature_names
expected_features_pm25_rest = pm25_model_t12.get_booster().feature_names
expected_features_pm10 = pm10_model_t.get_booster().feature_names


@router.get("/forecaster/global")
def get_global_shap(target: str,horizon: str):

    if target == "pm25":
        if horizon == "t":
            shap_importance = pm25_global_shap_t
        elif horizon == "t12":
            shap_importance = pm25_global_shap_t12
        elif horizon == "t24":
            shap_importance = pm25_global_shap_t24
        elif horizon == "t48":
            shap_importance = pm25_global_shap_t48
        else:
            raise HTTPException(
                status_code=400,
                detail="horizon must be 't', 't12', 't24', or 't48'"
            )

    elif target == "pm10":
        if horizon == "t":
            shap_importance = pm10_global_shap_t
        elif horizon == "t12":
            shap_importance = pm10_global_shap_t12
        elif horizon == "t24":
            shap_importance = pm10_global_shap_t24
        elif horizon == "t48":
            shap_importance = pm10_global_shap_t48
        else:
            raise HTTPException(
                status_code=400,
                detail="horizon must be 't', 't12', 't24', or 't48'"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="target must be 'pm25' or 'pm10'"
        )
    
    data = sorted(
    shap_importance.items(),
    key=lambda x: x[1],
    reverse=True
    )
    return {
        "target": target,
        "horizon": horizon,
        "data": [
            {
                "feature": feature,
                "importance": float(importance)
            }
            for feature, importance in data
        ]
    }
  
    
@router.post("/forecaster/local/pm25/auto")
def local_shap_pm25_auto(request: auto_input, horizon: str):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")

    pm25_X = prepare_input_pm25_auto(request, target_time=start_time, now_time=now_time)
    pm25_X12 = prepare_input_pm25_auto(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm25_X24 = prepare_input_pm25_auto(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm25_X48 = prepare_input_pm25_auto(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    
    pm25_X = pm25_X[expected_features_pm25_t]
    pm25_X12 = pm25_X12[expected_features_pm25_rest]
    pm25_X24 = pm25_X24[expected_features_pm25_rest]
    pm25_X48 = pm25_X48[expected_features_pm25_rest]
    
    prediction = None
    shap_result = None
    shap_values = None
    base_value = None
    if horizon == "t":
        X=pm25_X
        prediction = pm25_model_t.predict(pm25_X)
        shap_result = pm25_explainer_t(pm25_X)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t12":
        X=pm25_X12
        prediction = pm25_model_t12.predict(pm25_X12)
        shap_result = pm25_explainer_t12(pm25_X12)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t24":
        X=pm25_X24
        prediction = pm25_model_t24.predict(pm25_X24)
        shap_result = pm25_explainer_t24(pm25_X24)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t48":
        X=pm25_X48
        prediction = pm25_model_t48.predict(pm25_X48)
        shap_result = pm25_explainer_t48(pm25_X48)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]


    df = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].values,
        "shap_value": shap_values
    })

    df["abs_shap"] = df["shap_value"].abs()

    df["impact"] = np.where(
        df["shap_value"] >= 0,
        "increases_prediction",
        "decreases_prediction"
    )

    df = df.sort_values(
        "abs_shap",
        ascending=False
    )
    print("NaN values:")
    print(df[df.isna().any(axis=1)])

    print("Prediction:", prediction)
    print("Base:", base_value)
    print("SHAP NaN:", np.isnan(shap_values).any())
    return {
        "target": "pm25",
        "horizon": horizon,
        "prediction": float(prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }


@router.post("/predictor/local/pm25/manual")
def local_shap_pm25_manual(request: manual_input, horizon: str):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")

    pm25_X = prepare_input_pm25_manual(request, target_time=start_time, now_time=now_time)
    pm25_X12 = prepare_input_pm25_manual(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm25_X24 = prepare_input_pm25_manual(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm25_X48 = prepare_input_pm25_manual(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    
    pm25_X = pm25_X[expected_features_pm25_t]
    pm25_X12 = pm25_X12[expected_features_pm25_rest]
    pm25_X24 = pm25_X24[expected_features_pm25_rest]
    pm25_X48 = pm25_X48[expected_features_pm25_rest]
    
    prediction = None
    shap_result = None
    shap_values = None
    base_value = None
    if horizon == "t":
        X=pm25_X
        prediction = pm25_model_t.predict(pm25_X)
        shap_result = pm25_explainer_t(pm25_X)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t12":
        X=pm25_X12
        prediction = pm25_model_t12.predict(pm25_X12)
        shap_result = pm25_explainer_t12(pm25_X12)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t24":
        X=pm25_X24
        prediction = pm25_model_t24.predict(pm25_X24)
        shap_result = pm25_explainer_t24(pm25_X24)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t48":
        X=pm25_X48
        prediction = pm25_model_t48.predict(pm25_X48)
        shap_result = pm25_explainer_t48(pm25_X48)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]

    print("NaN in X:")
    print(X[X.isna().any(axis=1)])
    df = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].values,
        "shap_value": shap_values
    })

    df["abs_shap"] = df["shap_value"].abs()

    df["impact"] = np.where(
        df["shap_value"] >= 0,
        "increases_prediction",
        "decreases_prediction"
    )

    df = df.sort_values(
        "abs_shap",
        ascending=False
    )

    return {
        "target": "pm25",
        "prediction": float(prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }

@router.post("/forecaster/local/pm10/auto")
def local_shap_pm10_auto(request: auto_input, horizon: str):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")

    pm10_X = prepare_input_pm10_auto(request, target_time=start_time, now_time=now_time)
    pm10_X12 = prepare_input_pm10_auto(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm10_X24 = prepare_input_pm10_auto(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm10_X48 = prepare_input_pm10_auto(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    
    pm10_X = pm10_X[expected_features_pm10]
    pm10_X12 = pm10_X12[expected_features_pm10]
    pm10_X24 = pm10_X24[expected_features_pm10]
    pm10_X48 = pm10_X48[expected_features_pm10]
    
    prediction = None
    shap_result = None
    shap_values = None
    base_value = None
    if horizon == "t":
        X=pm10_X
        prediction = pm10_model_t.predict(pm10_X)
        shap_result = pm10_explainer_t(pm10_X)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t12":
        X=pm10_X12
        prediction = pm10_model_t12.predict(pm10_X12)
        shap_result = pm10_explainer_t12(pm10_X12)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t24":
        X=pm10_X24
        prediction = pm10_model_t24.predict(pm10_X24)
        shap_result = pm10_explainer_t24(pm10_X24)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t48":
        X=pm10_X48
        prediction = pm10_model_t48.predict(pm10_X48)
        shap_result = pm10_explainer_t48(pm10_X48)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]


    df = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].values,
        "shap_value": shap_values
    })

    df["abs_shap"] = df["shap_value"].abs()

    df["impact"] = np.where(
        df["shap_value"] >= 0,
        "increases_prediction",
        "decreases_prediction"
    )

    df = df.sort_values(
        "abs_shap",
        ascending=False
    )

    return {
        "target": "pm10",
        "horizon": horizon,
        "prediction": float(prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }
@router.post("/forecaster/local/pm10/manual")
def local_shap_pm10_manual(request: manual_input, horizon: str):
    now_time = pd.Timestamp.now(tz=IST).tz_localize(None).floor("6h")
    start_time = pd.Timestamp(request.target_time)
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(IST).tz_localize(None)
    start_time = start_time.floor("6h")

    pm10_X = prepare_input_pm10_manual(request, target_time=start_time, now_time=now_time)
    pm10_X12 = prepare_input_pm10_manual(request, target_time=start_time + pd.Timedelta(hours=12), now_time=now_time)
    pm10_X24 = prepare_input_pm10_manual(request, target_time=start_time + pd.Timedelta(hours=24), now_time=now_time)
    pm10_X48 = prepare_input_pm10_manual(request, target_time=start_time + pd.Timedelta(hours=48), now_time=now_time)
    
    pm10_X = pm10_X[expected_features_pm10]
    pm10_X12 = pm10_X12[expected_features_pm10]
    pm10_X24 = pm10_X24[expected_features_pm10]
    pm10_X48 = pm10_X48[expected_features_pm10]
    
    prediction = None
    shap_result = None
    shap_values = None
    base_value = None
    if horizon == "t":
        X=pm10_X
        prediction = pm10_model_t.predict(pm10_X)
        shap_result = pm10_explainer_t(pm10_X)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t12":
        X=pm10_X12
        prediction = pm10_model_t12.predict(pm10_X12)
        shap_result = pm10_explainer_t12(pm10_X12)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t24":
        X=pm10_X24
        prediction = pm10_model_t24.predict(pm10_X24)
        shap_result = pm10_explainer_t24(pm10_X24)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]
    elif horizon == "t48":
        X=pm10_X48
        prediction = pm10_model_t48.predict(pm10_X48)
        shap_result = pm10_explainer_t48(pm10_X48)
        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]


    df = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].values,
        "shap_value": shap_values
    })

    df["abs_shap"] = df["shap_value"].abs()

    df["impact"] = np.where(
        df["shap_value"] >= 0,
        "increases_prediction",
        "decreases_prediction"
    )

    df = df.sort_values(
        "abs_shap",
        ascending=False
    )

    return {
        "target": "pm10",
        "horizon": horizon,
        "prediction": float(prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }



@router.post("/forecaster/local/reasoning")
def local_reasoning(request: ForecasterReasoning, horizon: str):
    """
    Endpoint to generate reasoning for a local prediction based on SHAP values.
    """
    try:
        reasoning = generate_reasoning(request.model_dump(), horizon)
        return reasoning
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating reasoning: {str(e)}"
        )