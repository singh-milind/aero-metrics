import pandas as pd
import numpy as np
from src.feature_engineering.city_to_region import CITY_TO_REGION
from src.api.schemas.forecaster.auto.weather_features import add_weather_features
from src.api.schemas.forecaster.auto.pm_features import add_pm_features

def compute_features(df, logger, target_time):
    try:
        logger.info("Starting feature computation...")
        df = df.copy()
        
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["month"] = df["time"].dt.month
        df = add_weather_features(df)
        df=df[df['time'] == target_time]
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


        df["season_region"] = df["regional_season"].astype(str) + "_" + df["region"].astype(str)
        df.drop(columns=["regional_season", "region"], inplace=True)


        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        df["day_of_week"] = df["day_of_week"].map(day_map)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop(columns=["day_of_week"], inplace=True)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop(columns=["month"], inplace=True)
        
        
        time_map = {6: "Morning", 12: "Afternoon", 18: "Evening", 0: "Midnight"}
        df["time_of_day"] = df["hour"].map(time_map)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df.drop(columns=["hour"], inplace=True)
        
        categorical_cols = ["city", "weather_verdict", "time_of_day", "season_region"]


        df[categorical_cols] = df[categorical_cols].astype("category")

        for col in categorical_cols:
            if len(df[col].cat.categories) == 0:
                raise ValueError(f"No category available for {col}: {df[col].tolist()}")

        numeric_cols = [col for col in df.columns if col not in categorical_cols]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(float)


        df[categorical_cols] = df[categorical_cols].astype("category")
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(float)


        logger.info("Feature computation completed successfully.")
        categorical_cols = ["city", "weather_verdict", "time_of_day", "season_region"]
        df[categorical_cols] = df[categorical_cols].astype("category")
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        logger.exception("Error during feature computation.")
        raise