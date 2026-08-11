from src.model_building.predictor.pm10_model.runtime_features import build_more_features as pm10_build_more_features
from src.model_building.predictor.pm25_model.runtime_features import build_more_features as pm25_build_more_features
from src.model_building.predictor.pm10_model.split_data import split_data
from src.model_building.predictor.pm10_model.pm10_predictor import load_data
from src.utils.logger import get_logger
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[4]
logger = get_logger("pm10_evaluator")

def main():
    df = load_data(logger)
    X10,Y10= pm10_build_more_features(df,logger)
    X25,Y25= pm25_build_more_features(df,logger)
    X25_train, X25_test, y25_train, y25_test = split_data(X25, Y25, logger)
    X10_train, X10_test, y10_train, y10_test = split_data(X10, Y10, logger)
    model10 = joblib.load(ROOT_DIR / "src" / "model_building" / "predictor" / "pm10_model" / "pm10_predictor.pkl")
    model25 = joblib.load(ROOT_DIR / "src" / "model_building" / "predictor" / "pm25_model" / "pm25_predictor.pkl")

    # predictions = model25.predict(X25_train) * model10.predict(X10_train)
    predictions = model25.predict(X25_test)
    
    # r2 = r2_score(y10_test, predictions)
    # mae = mean_absolute_error(y10_test, predictions)
    # rmse = np.sqrt(mean_squared_error(y10_test, predictions))
    r2 = r2_score(y25_test, predictions)
    mae = mean_absolute_error(y25_test, predictions)
    rmse = np.sqrt(mean_squared_error(y25_test, predictions))

    logger.info(f"Model Performance:")
    logger.info(f"R²: {r2}")
    logger.info(f"MAE: {mae}")
    logger.info(f"RMSE: {rmse}")

if __name__ == "__main__":
    main()
