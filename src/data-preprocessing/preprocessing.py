from asyncio.log import logger
import pandas as pd
import numpy as np
import logging
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing.log"),
        logging.StreamHandler()
    ]
)


RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

try:
    aqi_df = pd.read_csv(RAW_DATA_DIR / "aqi_data.csv")
    weather_df = pd.read_csv(RAW_DATA_DIR / "weather_data.csv")
    logger.info("Raw datasets loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Missing input file: {e.filename}")
    raise
except Exception as e:
    logger.exception(f"Failed to load datasets: {e}")
    raise


def merge_datasets(aqi_df, weather_df):
    logger.info("Merging AQI and weather datasets...")

    try:
        merged_df = pd.merge(
            aqi_df,
            weather_df,
            on=["city", "time"],
            how="inner"
        )

        logger.info(f"Merged dataset shape: {merged_df.shape}")

        return merged_df

    except KeyError as e:
        logger.error(f"Missing merge column: {e}")
        raise

    except Exception as e:
        logger.exception(f"Dataset merge failed: {e}")
        raise

merged_df = merge_datasets(aqi_df, weather_df)

INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    merged_df.to_csv(
        INTERIM_DATA_DIR / "merged_data.csv",
        index=False
    )
    logger.info("Merged dataset saved successfully.")
except Exception as e:
    logger.exception(f"Failed to save merged dataset: {e}")
    raise


logger.info(f"AQI rows: {len(aqi_df):,}")
logger.info(f"Weather rows: {len(weather_df):,}")
logger.info(f"Merged rows: {len(merged_df):,}")
