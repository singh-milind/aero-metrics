import json
from fastapi import APIRouter
from src.utils.blob_storage import download_blob_bytes

router = APIRouter()


def load_metrics(blob_name: str):
    data = download_blob_bytes("metrics", blob_name)
    return json.loads(data.decode("utf-8"))


@router.post("/predcitor")
def get_metrics(model: str):
    if model == "pm25":
        return load_metrics(
            "predictor/pm25/predictor_model_metrics.json"
        )

    elif model == "pm10":
        return load_metrics(
            "predictor/pm10/predictor_model_metrics.json"
        )

    else:
        return {"error": "Model not found"}


@router.post("/forecaster")
def get_metrics(model: str, horizon: str):
    if model == "pm25":
        return load_metrics(
            f"forecaster/pm25_model/{horizon}_model_metrics.json"
        )

    elif model == "pm10":
        return load_metrics(
            f"forecaster/pm10_model/{horizon}_model_metrics.json"
        )

    else:
        return {"error": "Model not found"}