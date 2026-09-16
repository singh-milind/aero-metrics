import logging
import joblib
import pandas as pd
from pathlib import Path
import shap
from src.utils.logger import get_logger
from src.utils.blob_storage import load_csv
from src.utils.blob_storage import upload_model
from src.model_building.predictor.pm25_model.runtime_features import build_more_features
from src.model_building.predictor.pm25_model.split_data import split_data
from src.model_building.predictor.pm25_model.train_model import train_model



ROOT_DIR = Path(__file__).resolve().parents[4]
logger = get_logger("pm25_predictor")


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
        blob_name="predictor/pm25/pm25_predictor.pkl",
        model=model,
        artifact_name="pm25_predictor",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="predictor/pm25/pm25_explainer.pkl",
        model=explainer,
        artifact_name="pm25_explainer",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="predictor/pm25/pm25_global_shap_importance.pkl",
        model=global_shap_importance,
        artifact_name="pm25_global_shap_importance",
        logger=logger
    )
    upload_model(
        container_name="models",
        blob_name="predictor/pm25/pm25_global_shap_values.pkl",
        model=global_shap_values,
        logger=logger,
        artifact_name="pm25_global_shap_values"
    )
    upload_model(
        container_name="models",
        blob_name="predictor/pm25/pm25_global_shap_feature_values.pkl",
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
if __name__ == "__main__":
    main()
    