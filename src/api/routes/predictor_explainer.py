import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.schemas.predictor.predictor import PredictorInput,prepare_input
from src.api.schemas.predictor.reasoning import PredictorReasoning
from src.api.services.predictor_reasoning import generate_reasoning

router = APIRouter()

pm25_global_shap = joblib.load("models/predictor/pm25/pm25_global_shap.pkl")
pm10_global_shap = joblib.load("models/predictor/pm10/pm10_global_shap.pkl")

pm25_explainer = joblib.load("models/predictor/pm25/pm25_explainer.pkl")
pm10_explainer = joblib.load("models/predictor/pm10/pm10_explainer.pkl")

pm25_model = joblib.load("models/predictor/pm25/pm25_predictor.pkl")
pm10_model = joblib.load("models/predictor/pm10/pm10_predictor.pkl")

expected_features_pm25 = pm25_model.get_booster().feature_names
expected_features_pm10 = pm10_model.get_booster().feature_names


@router.get("/predictor/global")
def get_global_shap(target: str):

    if target == "pm25":
        shap_importance = pm25_global_shap

    elif target == "pm10":
        shap_importance = pm10_global_shap

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
        "data": [
            {
                "feature": feature,
                "importance": float(importance)
            }
            for feature, importance in data
        ]
    }
  
    
@router.post("/predictor/local/pm25")
def local_shap_pm25(request: PredictorInput):

    X = prepare_input(request)
    X = X[expected_features_pm25]

    prediction = pm25_model.predict(X)

    shap_result = pm25_explainer(X)

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
        "target": "pm25",
        "prediction": float(prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }


@router.post("/predictor/local/pm10")
def local_shap_pm10(request: PredictorInput):

    X = prepare_input(request)

    X = X[expected_features_pm10]

    ratio_prediction = pm10_model.predict(X)

    shap_result = pm10_explainer(X)

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
        "increases_ratio",
        "decreases_ratio"
    )

    df = df.sort_values(
        "abs_shap",
        ascending=False
    )

    return {
        "target": "pm10_ratio",
        "ratio_prediction": float(ratio_prediction[0]),
        "base_value": float(base_value),
        "data": df.to_dict(orient="records")
    }
    
@router.post("/predictor/local/reasoning")
def local_reasoning(request: PredictorReasoning):
    """
    Endpoint to generate reasoning for a local prediction based on SHAP values.
    """
    try:
        reasoning = generate_reasoning(request.model_dump())
        return reasoning
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating reasoning: {str(e)}"
        )