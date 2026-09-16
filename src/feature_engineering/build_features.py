
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineering.weather_features import apply_weather_verdict
from src.feature_engineering.region import region_map
from src.feature_engineering.temporal_features import apply_regional_season,add_time_features
from src.database.ingest_historical_data import main as ingest_historical_data
from src.utils.logger import get_logger
from src.utils.blob_storage import upload_csv,load_csv

ROOT_DIR = Path(__file__).resolve().parents[2]

logger = get_logger("feature_engineering")

def load_data(logger):

    try:
        df = load_csv(
            container_name="data",
            blob_name="interim/merged_data.csv"
        )
        logger.info("Merged dataset loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Missing input file: {e.filename}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load datasets: {e}")
        raise  
    return df

def save_data(df, logger):
    csv_data = df.to_csv(index=False)

    try:
        upload_csv(
            container_name="data",
            blob_name="processed/engineered_features.csv",
            csv_data=csv_data
        )
        logger.info("Engineered features saved successfully.")
    except Exception as e:
        logger.exception(f"Failed to save engineered features: {e}")
        raise


def main():
        try:
            df = load_data(logger)
            logger.info(f"Loaded {len(df):,} rows.")
            df = add_time_features(df, logger)
            df = apply_weather_verdict(df, logger)
            df = region_map(df, logger)
            df = apply_regional_season(df, logger)
            save_data(df, logger)
            logger.info(f"Feature engineering completed. Final dataset has {len(df):,} rows.")
            ingest_historical_data()
        except FileNotFoundError as e:
            logger.error(f"File not found: {e.filename}. Ensure that the data gathering step has been completed.")
        except Exception:
            logger.exception("Feature engineering pipeline failed.")
            raise



if __name__ == "__main__":
    main()