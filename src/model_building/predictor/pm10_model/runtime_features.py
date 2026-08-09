import pandas as pd
import numpy as np



def build_more_features(df,logger):
    try:
        logger.info("Starting feature engineering...")

        columns_to_drop = [ "pm2_5", "pm10", "time"]

        X = df.drop(columns=columns_to_drop)
        Y = df['pm10']/df['pm2_5']

        # Feature Interactions

        logger.info("Creating interaction features...")

        X["season_region"] = X["regional_season"].astype(str) + "_" + X["region"].astype(str)
        #X["season_city"] = X["regional_season"].astype(str) + "_" + X["city"].astype(str)
        X.drop(columns=["regional_season", "region"], inplace=True)

        X["temp_humidity"] = (
            X["temperature_2m"] * X["relative_humidity_2m"]
        )

        X["wind_precip"] = (
            X["wind_speed_10m"] * X["precipitation"]
        )

        X["pressure_temp"] = (
            X["surface_pressure"] * X["temperature_2m"]
        )

        # Wind Direction

        logger.info("Encoding wind direction...")

        X["wind_dir_sin"] = np.sin(np.deg2rad(X["wind_direction_10m"]))
        X["wind_dir_cos"] = np.cos(np.deg2rad(X["wind_direction_10m"]))

        X.drop(columns=["wind_direction_10m"], inplace=True)

        # Cyclic Features
        day_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }

        X["day_of_week"] = X["day_of_week"].map(day_map)
        logger.info("Encoding cyclic time features...")

        X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        X.drop(columns=["day_of_week"], inplace=True)

        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        X.drop(columns=["month"], inplace=True)

        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        X.drop(columns=["hour"], inplace=True)

        # Convert categorical columns

        logger.info("Converting categorical columns...")

        categorical_cols = [
            "city",
            "weather_verdict",
            "time_of_day",
            "season_region",
            # "season_city",
        ]

        X[categorical_cols] = X[categorical_cols].astype("category")

        logger.info("Feature engineering completed successfully.")
        return X, Y

    except Exception as e:
        logger.exception("Error during feature engineering.")
        raise