import pandas as pd
import numpy as np
from sqlalchemy import text

from src.database.connection import engine
from pathlib import Path
from src.utils.logger import get_logger
logger=get_logger("historical_data_ingest")

def clear_historical_data():
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM historical_data"))

    logger.info("Deleted all existing historical data.")

def calculate_sub_index(C, breakpoints):
    """
    Calculate pollutant sub-index using linear interpolation.

    C = pollutant concentration
    breakpoints = [(BLO, BHI, ILO, IHI), ...]
    """

    if pd.isna(C):
        return np.nan

    for BLO, BHI, ILO, IHI in breakpoints:
        if BLO <= C <= BHI:
            return (
                ((IHI - ILO) / (BHI - BLO))
                * (C - BLO)
                + ILO
            )

    return np.nan


# PM2.5 breakpoints (µg/m³)
pm25_bp = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 1000, 400, 500),
]


# PM10 breakpoints (µg/m³)
pm10_bp = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 250, 100, 200),
    (250, 350, 200, 300),
    (350, 430, 300, 400),
    (430, 1000, 400, 500),
]


def calculate_aqi(row):
    pm25_index = calculate_sub_index(
        row["pm2_5"],
        pm25_bp
    )

    pm10_index = calculate_sub_index(
        row["pm10"],
        pm10_bp
    )

    return np.nanmax([
        pm25_index,
        pm10_index
    ])
def ingest(df):
    logger.info("Ingesting historical data...")
    df = df.copy()
    day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

    df["day_of_week"] = df["day_of_week"].map(day_map)
    # Ensure correct datetime type
    df["time"] = pd.to_datetime(df["time"])

    columns = [
        "time",
        "city",
        "pm10",
        "pm2_5",
        "aqi",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "surface_pressure",
        "month",
        "hour",
        "day_of_week",
        "time_of_day",
        "is_weekend",
        "weather_verdict",
        "region",
        "regional_season",
    ]
    df["aqi"] = df.apply(calculate_aqi, axis=1)
    df = df[columns]

    df = df.dropna(subset=["time", "city"])
    df = df.drop_duplicates(subset=["city", "time"])

    df.to_sql(
        "historical_data",
        con=engine,
        schema="public",
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi",
    )

    print(f"Inserted {len(df)} historical rows.")


def main():
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data" / "processed"
    df = pd.read_csv(DATA_DIR / "engineered_features.csv")
    clear_historical_data()
    ingest(df)
    logger.info("Historical data ingestion completed.")
    
if __name__ == "__main__":
    main()