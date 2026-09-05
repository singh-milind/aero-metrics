import json
from fastapi import APIRouter
router = APIRouter()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


@router.post("/predcitor")
def get_metrics(model: str):
    if model == "pm25":
        metrics_path = BASE_DIR / "metrics" / "predictor" / "pm25" /"predictor_model_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    elif model == "pm10":
        metrics_path = BASE_DIR / "metrics" / "predictor" / "pm10" /"predictor_model_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    else:
        return {"error": "Model not found"}
    
    
@router.post("/forecaster")
def get_metrics(model: str, horizon: str):
    if model == "pm25":
        metrics_path = BASE_DIR / "metrics" / "forecaster" / "pm25_model" / f"{horizon}_model_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    elif model == "pm10":
        metrics_path = BASE_DIR / "metrics" / "forecaster" / "pm10_model" / f"{horizon}_model_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    else:
        return {"error": "Model not found"}
