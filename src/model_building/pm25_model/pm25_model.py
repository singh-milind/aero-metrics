import logging

import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.model_building.pm25_model.build_more_features import build_more_features
from src.model_building.pm25_model.split_data import split_data
from src.model_building.pm25_model.train_model import train_model



ROOT_DIR = Path(__file__).resolve().parents[3]
logger = get_logger("pm25_model")


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
    X,Y= build_more_features(df,logger)
    X_train, X_test, y_train, y_test = split_data(X, Y, logger)
    model = train_model(X_train, y_train, logger)

if __name__ == "__main__":
    main()
    