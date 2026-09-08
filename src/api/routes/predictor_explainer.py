import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from io import BytesIO
from src.utils.blob_storage import download_blob_bytes
from src.api.schemas.predictor.predictor import PredictorInput,prepare_input
from src.api.schemas.predictor.reasoning import PredictorReasoning
from src.api.services.predictor_reasoning import generate_reasoning

router = APIRouter()

pm25_global_shap = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm25/pm25_global_shap.pkl")))
pm10_global_shap = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm10/pm10_global_shap.pkl")))

pm25_global_shap_values = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm25/pm25_global_shap_values.pkl")))
pm25_global_shap_features = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm25/pm25_global_shap_feature_values.pkl")))

pm10_global_shap_values = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm10/pm10_global_shap_values.pkl")))
pm10_global_shap_features = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm10/pm10_global_shap_feature_values.pkl")))

pm25_explainer = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm25/pm25_explainer.pkl")))
pm10_explainer = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm10/pm10_explainer.pkl")))

pm25_model = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm25/pm25_predictor.pkl")))
pm10_model = joblib.load(BytesIO(download_blob_bytes("models", "predictor/pm10/pm10_predictor.pkl")))

expected_features_pm25 = pm25_model.get_booster().feature_names
expected_features_pm10 = pm10_model.get_booster().feature_names

@router.get("/predictor/global")
def get_global_shap(target: str):
    if target == "pm25":
        importance = pm25_global_shap
        shap_raw = pm25_global_shap_values
        feature_values_raw = pm25_global_shap_features
    elif target == "pm10":
        importance = pm10_global_shap
        shap_raw = pm10_global_shap_values
        feature_values_raw = pm10_global_shap_features
    else:
        raise HTTPException(
            status_code=400,
            detail="target must be 'pm25' or 'pm10'"
        )

    # Extract numeric values if artifact is a SHAP Explanation
    if hasattr(shap_raw, "values"):
        shap_source = shap_raw.values
    else:
        shap_source = shap_raw

    TOP_FEATURES = 20
    MAX_SAMPLES = 3000

    shap_rows = len(shap_source)
    feature_rows = len(feature_values_raw)

    if shap_rows != feature_rows:
        raise HTTPException(
            status_code=500,
            detail=(
                f"SHAP/features row mismatch: "
                f"{shap_rows} vs {feature_rows}"
            )
        )

    # Global feature importance
    importance_data = sorted(
        importance.items(),
        key=lambda x: float(x[1]),
        reverse=True
    )

    # Top features for beeswarm
    top_features = [
        str(feature)
        for feature, _ in importance_data[:TOP_FEATURES]
    ]

    # Feature names
    feature_names = [
        str(feature)
        for feature in feature_values_raw.columns
    ]

    # Match top features to columns
    selected_indices = [
        feature_names.index(feature)
        for feature in top_features
        if feature in feature_names
    ]

    selected_features = [
        feature_names[i]
        for i in selected_indices
    ]

    # Sample BEFORE converting the entire SHAP matrix
    if shap_rows > MAX_SAMPLES:
        rng = np.random.default_rng(42)

        indices = rng.choice(
            shap_rows,
            MAX_SAMPLES,
            replace=False
        )

        indices.sort()
    else:
        indices = np.arange(shap_rows)

    shap_sample = shap_source[indices]
    feature_sample = feature_values_raw.iloc[indices]

    # Convert only sampled SHAP values
    shap_values = np.asarray(
        shap_sample,
        dtype=np.float32
    )

    # Select top features
    selected_shap_values = shap_values[
        :,
        selected_indices
    ]

    # Select matching feature values
    selected_feature_values = (
        feature_sample
        .iloc[:, selected_indices]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=np.float32
        )
    )

    # Final validation
    if selected_shap_values.shape != selected_feature_values.shape:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Beeswarm shape mismatch: "
                f"SHAP={selected_shap_values.shape}, "
                f"features={selected_feature_values.shape}"
            )
        )

    return {
        "target": target,
        "data": [
            {
                "feature": str(feature),
                "importance": float(value)
            }
            for feature, value in importance_data
        ],
        "beeswarm": {
            "feature_names": selected_features,
            "shap_values": selected_shap_values.tolist(),
            "feature_values": selected_feature_values.tolist()
        }
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