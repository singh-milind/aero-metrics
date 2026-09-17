import logging

import joblib
import pandas as pd
from pathlib import Path

import shap
from src.utils.logger import get_logger
from src.model_building.forecaster.pm25_model.t_model.runtime_features import build_more_features
from src.model_building.forecaster.pm25_model.t_model.split_data import split_data
from src.model_building.forecaster.pm25_model.t_model.train_model import train_model
from src.utils.blob_storage import load_csv
from src.utils.blob_storage import upload_model



ROOT_DIR = Path(__file__).resolve().parents[5]
logger = get_logger("pm25_forecaster_t")


def load_data(logger):
    try:
        df = load_csv(
            container_name="data",
            blob_name="processed/engineered_features.csv"
        )
        logger.info("Engineered features loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Missing input file: {e.filename}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load datasets: {e}")
        raise
    return df

def upload_model_to_blob(model,explainer,global_shap_importance,global_shap_values,X_test, logger):

    upload_model(
        container_name="models",
        blob_name="forecaster/pm25/t_model/pm25_forecaster.pkl",
        model=model,
        artifact_name="pm25_forecaster",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="forecaster/pm25/t_model/pm25_explainer.pkl",
        model=explainer,
        artifact_name="pm25_explainer",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="forecaster/pm25/t_model/pm25_global_shap_importance.pkl",
        model=global_shap_importance,
        artifact_name="pm25_global_shap_importance",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="forecaster/pm25/t_model/pm25_global_shap_values.pkl",
        model=global_shap_values,
        logger=logger,
        artifact_name="pm25_global_shap_values"
    )
    upload_model(
        container_name="models",
        blob_name="forecaster/pm25/t_model/pm25_global_shap_feature_values.pkl",
        model=X_test,
        logger=logger,
        artifact_name="pm25_global_shap_feature_values"
    )
def main():
    df = load_data(logger)
    X, Y = build_more_features(df, logger)
    X_train, X_test, y_train, y_test = split_data(X, Y, logger)
    model = train_model(X_train, y_train, logger)

    explainer = shap.TreeExplainer(model)
    global_shap_values = explainer(X_test)
    global_shap_importance = dict(zip(X_test.columns, abs(global_shap_values.values).mean(axis=0)))

    upload_model_to_blob(model, explainer, global_shap_importance, global_shap_values, X_test, logger)
    model_dir = ROOT_DIR / "models" / "forecaster" / "pm25" / "t_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "pm25_forecaster_t.pkl")
    joblib.dump(explainer, model_dir / "pm25_explainer.pkl")
    joblib.dump(global_shap_importance, model_dir / "pm25_global_shap.pkl")
    joblib.dump(global_shap_values, model_dir / "pm25_global_shap_values.pkl")
    joblib.dump(X_test, model_dir / "pm25_global_shap_feature_values.pkl")
if __name__ == "__main__":
    main()
    