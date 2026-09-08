import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from src.utils.logger import get_logger

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
from io import BytesIO
from src.utils.blob_storage import download_blob_bytes
IST = ZoneInfo("Asia/Kolkata")
router = APIRouter()
logger=get_logger("forecaster_explainer")

pm25_global_shap_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t_model/pm25_global_shap.pkl")))
pm25_global_shap_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t12_model/pm25_global_shap.pkl")))
pm25_global_shap_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t24_model/pm25_global_shap.pkl")))
pm25_global_shap_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t48_model/pm25_global_shap.pkl")))

pm25_shap_values_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t_model/pm25_global_shap_values.pkl")))
pm25_shap_values_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t12_model/pm25_global_shap_values.pkl")))
pm25_shap_values_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t24_model/pm25_global_shap_values.pkl")))
pm25_shap_values_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t48_model/pm25_global_shap_values.pkl")))

pm25_shap_features_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t_model/pm25_global_shap_feature_values.pkl")))
pm25_shap_features_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t12_model/pm25_global_shap_feature_values.pkl")))
pm25_shap_features_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t24_model/pm25_global_shap_feature_values.pkl")))
pm25_shap_features_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t48_model/pm25_global_shap_feature_values.pkl")))

pm10_global_shap_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t_model/pm10_global_shap.pkl")))
pm10_global_shap_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t12_model/pm10_global_shap.pkl")))
pm10_global_shap_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t24_model/pm10_global_shap.pkl")))
pm10_global_shap_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t48_model/pm10_global_shap.pkl")))

pm10_shap_values_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t_model/pm10_global_shap_values.pkl")))
pm10_shap_values_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t12_model/pm10_global_shap_values.pkl")))
pm10_shap_values_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t24_model/pm10_global_shap_values.pkl")))
pm10_shap_values_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t48_model/pm10_global_shap_values.pkl")))

pm10_shap_features_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t_model/pm10_global_shap_feature_values.pkl")))
pm10_shap_features_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t12_model/pm10_global_shap_feature_values.pkl")))
pm10_shap_features_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t24_model/pm10_global_shap_feature_values.pkl")))
pm10_shap_features_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t48_model/pm10_global_shap_feature_values.pkl")))

pm10_explainer_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t_model/pm10_explainer.pkl")))
pm10_explainer_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t12_model/pm10_explainer.pkl")))
pm10_explainer_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t24_model/pm10_explainer.pkl")))
pm10_explainer_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t48_model/pm10_explainer.pkl")))

pm25_explainer_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t_model/pm25_explainer.pkl")))
pm25_explainer_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t12_model/pm25_explainer.pkl")))
pm25_explainer_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t24_model/pm25_explainer.pkl")))
pm25_explainer_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t48_model/pm25_explainer.pkl")))

pm25_model_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t_model/pm25_forecaster_t.pkl")))
pm25_model_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t12_model/pm25_forecaster_t12.pkl")))
pm25_model_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t24_model/pm25_forecaster_t24.pkl")))
pm25_model_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm25/t48_model/pm25_forecaster_t48.pkl")))

pm10_model_t = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t_model/pm10_forecaster_t.pkl")))
pm10_model_t12 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t12_model/pm10_forecaster_t12.pkl")))
pm10_model_t24 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t24_model/pm10_forecaster_t24.pkl")))
pm10_model_t48 = joblib.load(BytesIO(download_blob_bytes("models", "forecaster/pm10/t48_model/pm10_forecaster_t48.pkl")))


expected_features_pm25_t = pm25_model_t.get_booster().feature_names
expected_features_pm25_rest = pm25_model_t12.get_booster().feature_names
expected_features_pm10 = pm10_model_t.get_booster().feature_names


@router.get("/forecaster/global")
def get_global_shap(target: str, horizon: str):
    if horizon not in ["t", "t12", "t24", "t48"]:
        raise HTTPException(
            status_code=400,
            detail="horizon must be 't', 't12', 't24', or 't48'"
        )

    logger.info(
        f"Fetching global SHAP for target: {target}, horizon: {horizon}"
    )

    if target == "pm25":
        importance_map = {
            "t": pm25_global_shap_t,
            "t12": pm25_global_shap_t12,
            "t24": pm25_global_shap_t24,
            "t48": pm25_global_shap_t48,
        }
        shap_map = {
            "t": pm25_shap_values_t,
            "t12": pm25_shap_values_t12,
            "t24": pm25_shap_values_t24,
            "t48": pm25_shap_values_t48,
        }
        features_map = {
            "t": pm25_shap_features_t,
            "t12": pm25_shap_features_t12,
            "t24": pm25_shap_features_t24,
            "t48": pm25_shap_features_t48,
        }

    elif target == "pm10":
        importance_map = {
            "t": pm10_global_shap_t,
            "t12": pm10_global_shap_t12,
            "t24": pm10_global_shap_t24,
            "t48": pm10_global_shap_t48,
        }
        shap_map = {
            "t": pm10_shap_values_t,
            "t12": pm10_shap_values_t12,
            "t24": pm10_shap_values_t24,
            "t48": pm10_shap_values_t48,
        }
        features_map = {
            "t": pm10_shap_features_t,
            "t12": pm10_shap_features_t12,
            "t24": pm10_shap_features_t24,
            "t48": pm10_shap_features_t48,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="target must be 'pm25' or 'pm10'"
        )

    importance = importance_map[horizon]
    shap_raw = shap_map[horizon]
    feature_values_raw = features_map[horizon]

    logger.info("Selected SHAP artifacts")

    MAX_SAMPLES = 3000
    TOP_FEATURES = 20

    # Extract numeric SHAP values from SHAP Explanation
    if hasattr(shap_raw, "values"):
        shap_source = shap_raw.values
    else:
        shap_source = shap_raw

    logger.info(
        f"SHAP source type: {type(shap_source)}"
    )

    # Get dimensions
    shap_rows = len(shap_source)
    feature_rows = len(feature_values_raw)

    logger.info(
        f"SHAP rows: {shap_rows}, "
        f"Feature rows: {feature_rows}"
    )

    if shap_rows != feature_rows:
        raise HTTPException(
            status_code=500,
            detail=(
                f"SHAP/features row mismatch: "
                f"{shap_rows} vs {feature_rows}"
            )
        )

    # Global importance
    importance_data = sorted(
        importance.items(),
        key=lambda x: float(x[1]),
        reverse=True
    )

    top_features = [
        str(feature)
        for feature, _ in importance_data[:TOP_FEATURES]
    ]

    logger.info(
        f"Top features selected: {top_features}"
    )

    # Sample BEFORE converting the complete matrix
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

    logger.info(
        f"Sampling {len(indices)} SHAP rows"
    )

    # Slice numeric SHAP values
    shap_sample = shap_source[indices]

    # Slice matching feature rows
    feature_sample = feature_values_raw.iloc[indices]

    logger.info("Sample extracted")

    # Convert ONLY the sampled SHAP values
    shap_values = np.asarray(
        shap_sample,
        dtype=np.float32
    )

    logger.info(
        f"SHAP sample shape: {shap_values.shape}"
    )

    if shap_values.ndim != 2:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Expected 2D SHAP values, "
                f"got shape {shap_values.shape}"
            )
        )

    # Feature names
    feature_names = [
        str(feature)
        for feature in feature_sample.columns
    ]

    # Match top features to SHAP columns
    selected_indices = [
        feature_names.index(feature)
        for feature in top_features
        if feature in feature_names
    ]

    selected_features = [
        feature_names[i]
        for i in selected_indices
    ]

    logger.info(
        f"Selected {len(selected_features)} beeswarm features"
    )

    # Select top SHAP columns
    selected_shap = shap_values[
        :,
        selected_indices
    ]

    # Select corresponding feature values
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

    logger.info(
        f"Final beeswarm shapes: "
        f"SHAP={selected_shap.shape}, "
        f"features={selected_feature_values.shape}"
    )

    return {
        "target": target,
        "horizon": horizon,
        "data": [
            {
                "feature": str(feature),
                "importance": float(value)
            }
            for feature, value in importance_data
        ],
        "beeswarm": {
            "feature_names": selected_features,
            "shap_values": selected_shap.tolist(),
            "feature_values": selected_feature_values.tolist()
        }
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