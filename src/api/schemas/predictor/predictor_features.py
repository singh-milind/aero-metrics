import pandas as pd
import numpy as np
from src.feature_engineering.city_to_region import CITY_TO_REGION

def compute_features(df, logger):
    try:
        logger.info("Starting feature computation...")
        df = df.copy()

        df["region"] = df["city"].map(CITY_TO_REGION)
        if df["region"].isna().any():
            missing = df.loc[df["region"].isna(), "city"].unique()
            raise ValueError(f"Missing region mapping for: {missing}")

        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        def get_season(row):
            month, region = row["month"], row["region"]
            if region == "South":
                return "Winter" if month in [1, 2] else "Summer" if month in [3, 4, 5] else "Southwest Monsoon" if month in [6, 7, 8, 9] else "Retreating Monsoon"
            if region == "Northeast":
                return "Winter" if month in [12, 1, 2] else "Pre-Monsoon" if month in [3, 4] else "Extended Monsoon" if month in [5, 6, 7, 8, 9, 10] else "Post-Monsoon"
            if region in ["North", "Central", "East", "West", "NCR"]:
                return "Winter" if month in [12, 1, 2] else "Summer" if month in [3, 4, 5, 6] else "Southwest Monsoon" if month in [7, 8, 9] else "Post-Monsoon"
            if region == "UT":
                return "Winter" if month in [12, 1, 2] else "Spring" if month in [3, 4] else "Summer" if month in [5, 6, 7, 8] else "Autumn"
            return "Unknown"

        df["regional_season"] = df.apply(get_season, axis=1)

        def get_weather(row):
            temp = row["temperature_2m"]
            humidity = row["relative_humidity_2m"]
            wind = row["wind_speed_10m"]
            rain = row["precipitation"]
            if rain > 2.0:
                return "Rainy / Washed"
            if wind > 15.0:
                return "Windy & Clear"
            if temp < 18.0 and wind < 5.0:
                return "Cold & Stagnant"
            if temp > 35.0 and humidity < 40.0:
                return "Hot & Dry"
            if temp > 28.0 and humidity > 70.0:
                return "Hot & Humid"
            return "Pleasant / Normal"

        df["weather_verdict"] = df.apply(get_weather, axis=1)

        df["season_region"] = df["regional_season"].astype(str) + "_" + df["region"].astype(str)
        df.drop(columns=["regional_season", "region"], inplace=True)

        df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
        df["wind_precip"] = df["wind_speed_10m"] * df["precipitation"]
        df["pressure_temp"] = df["surface_pressure"] * df["temperature_2m"]

        df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))
        df.drop(columns=["wind_direction_10m"], inplace=True)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop(columns=["day_of_week"], inplace=True)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop(columns=["month"], inplace=True)
        
        time_map = {"Morning": 6, "Afternoon": 12, "Evening": 18, "Midnight": 0}
        df["hour"] = df["time_of_day"].map(time_map)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df.drop(columns=["hour"], inplace=True)

        categorical_cols = ["city", "weather_verdict", "time_of_day", "season_region"]
        df[categorical_cols] = df[categorical_cols].astype("category")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        logger.info("Feature computation completed successfully.")
        return df

    except Exception:
        logger.exception("Error during feature computation.")
        raise