import logging

import joblib
import pandas as pd
from pathlib import Path

import shap
from src.utils.logger import get_logger
from src.model_building.forecaster.pm10_model.t24_model.runtime_features import build_more_features
from src.model_building.forecaster.pm10_model.t24_model.split_data import split_data
from src.model_building.forecaster.pm10_model.t24_model.train_model import train_model



ROOT_DIR = Path(__file__).resolve().parents[5]
logger = get_logger("pm10_forecaster_t24")


def load_data(logger):
    PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "engineered_features.csv")
        logger.info("Engineered features loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Missing input file: {e.filename}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load datasets: {e}")
        raise
    return df


def main():
    df = load_data(logger)
    df = load_data(logger)
    X, Y = build_more_features(df, logger)
    X_train, X_test, y_train, y_test = split_data(X, Y, logger)
    model = train_model(X_train, y_train, logger)

    explainer = shap.TreeExplainer(model)
    global_shap_values = explainer(X_test)
    global_shap_importance = dict(zip(X_test.columns, abs(global_shap_values.values).mean(axis=0)))

    model_dir = ROOT_DIR / "src" / "model_building" / "forecaster" / "pm10_model" / "t24_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "pm10_forecaster_t24.pkl")
    joblib.dump(explainer, model_dir / "pm10_explainer.pkl")
    joblib.dump(global_shap_importance, model_dir / "pm10_global_shap.pkl")
    
if __name__ == "__main__":
    main()
    