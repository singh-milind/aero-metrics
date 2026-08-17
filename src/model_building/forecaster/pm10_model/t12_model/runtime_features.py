import numpy as np

def build_more_features(df, logger):
    try:
        logger.info("Starting feature engineering...")

        df = df.sort_values(["time", "city"]).copy()

        X = df.drop(
            columns=["pm2_5", "pm10", "time"]
        ).copy()
        
        # df["pm_ratio"] = df["pm10"] / df["pm2_5"]

        # Y = df["pm_ratio"].copy()
        
        Y = df.groupby("city")["pm10"].transform(
                lambda x: x.clip(
                    upper=x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
                )
            )

        Y = Y.groupby(df["city"]).shift(-2)

        logger.info("Adding Lag Features...")

        for hours in [12, 24, 48]:
            periods = hours // 6
            X[f"pm10_lag_{hours}h"] = (
                df.groupby("city")["pm10"]
                .shift(periods)
            )

        logger.info("Adding Rolling Mean Features...")

        past_pm10 = (
            df.groupby("city")["pm10"]
            .shift(1)
        )

        X["pm10_rolling_mean_24h"] = (
            past_pm10.groupby(df["city"])
                    .rolling(window=4)
                    .mean()
                    .reset_index(level=0, drop=True)
        )

        X["pm10_rolling_mean_48h"] = (
            past_pm10.groupby(df["city"])
                    .rolling(window=8)
                    .mean()
                    .reset_index(level=0, drop=True)
        )
        
        X["pm10_change_12h"] = (
            X["pm10_lag_12h"] -
            X["pm10_lag_24h"]
        )

        X["pm10_change_24h"] = (
            X["pm10_lag_24h"] -
            X["pm10_lag_48h"]
        )
        X["pm10_acceleration"] = (
        X["pm10_lag_12h"]
        - 2 * X["pm10_lag_24h"]
        + X["pm10_lag_48h"]
    )

        X["pm10_recent_max"] = X[
            ["pm10_lag_12h", "pm10_lag_24h", "pm10_lag_48h"]
        ].max(axis=1)

        X["pm10_recent_mean"] = X[
            ["pm10_lag_12h", "pm10_lag_24h", "pm10_lag_48h"]
        ].mean(axis=1)
        weather_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure",
            "precipitation"
            ]

        for col in weather_cols:
            X[f"{col}_change_6h"] = (
                df.groupby("city")[col].diff(1)
            )
            
        for col in weather_cols:
            X[f"{col}_change_12h"] = (
                df.groupby("city")[col].diff(2)
            )
            
        for col in weather_cols:
            X[f"{col}_change_24h"] = (
                df.groupby("city")[col].diff(4)
            )
            
        for col in weather_cols:
            X[f"{col}_change_48h"] = (
                df.groupby("city")[col].diff(8)
            )
        temporal_cols = [
            "pm10_lag_12h",
            "pm10_lag_24h",
            "pm10_lag_48h",
            "pm10_rolling_mean_24h",
            "pm10_rolling_mean_48h",
            "pm10_change_12h",
            "pm10_change_24h",
            "pm10_acceleration",
            "pm10_recent_max",
            "pm10_recent_mean",
            
            "temperature_2m_change_6h",
            "relative_humidity_2m_change_6h",
            "wind_speed_10m_change_6h",
            "surface_pressure_change_6h",
            "precipitation_change_6h",
           "temperature_2m_change_12h",
            "relative_humidity_2m_change_12h",
            "wind_speed_10m_change_12h",
            "surface_pressure_change_12h",
            "precipitation_change_12h",
            "temperature_2m_change_24h",
            "relative_humidity_2m_change_24h",
            "wind_speed_10m_change_24h",
            "surface_pressure_change_24h",
            "precipitation_change_24h",
            "temperature_2m_change_48h",
            "relative_humidity_2m_change_48h",
            "wind_speed_10m_change_48h",
            "surface_pressure_change_48h",
            "precipitation_change_48h"
        ]

        valid = X[temporal_cols].notna().all(axis=1) & Y.notna()

        X = X.loc[valid].copy()
        Y = Y.loc[valid].copy()
        

        X["stagnation_index"] = (
        X["relative_humidity_2m"]
        / (X["wind_speed_10m"] + 1)
    )

        X["dry_stagnation"] = (
            X["relative_humidity_2m"]
            * (1 / (X["wind_speed_10m"] + 1))
        )

        X["no_rain"] = (
            X["precipitation"] == 0
        ).astype(int)
        
        
        logger.info("Creating interaction features...")

        X["season_region"] = (
            X["regional_season"].astype(str)
            + "_"
            + X["region"].astype(str)
        )

        X.drop(
            columns=["regional_season", "region"],
            inplace=True
        )

        X["temp_humidity"] = (
            X["temperature_2m"]
            * X["relative_humidity_2m"]
        )

        X["wind_precip"] = (
            X["wind_speed_10m"]
            * X["precipitation"]
        )

        X["pressure_temp"] = (
            X["surface_pressure"]
            * X["temperature_2m"]
        )

        logger.info("Encoding wind direction...")

        X["wind_dir_sin"] = np.sin(
            np.deg2rad(X["wind_direction_10m"])
        )

        X["wind_dir_cos"] = np.cos(
            np.deg2rad(X["wind_direction_10m"])
        )

        X.drop(
            columns=["wind_direction_10m"],
            inplace=True
        )

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

        X["dow_sin"] = np.sin(
            2 * np.pi * X["day_of_week"] / 7
        )

        X["dow_cos"] = np.cos(
            2 * np.pi * X["day_of_week"] / 7
        )

        X.drop(columns=["day_of_week"], inplace=True)

        X["month_sin"] = np.sin(
            2 * np.pi * X["month"] / 12
        )

        X["month_cos"] = np.cos(
            2 * np.pi * X["month"] / 12
        )

        X.drop(columns=["month"], inplace=True)

        X["hour_sin"] = np.sin(
            2 * np.pi * X["hour"] / 24
        )

        X["hour_cos"] = np.cos(
            2 * np.pi * X["hour"] / 24
        )

        X.drop(columns=["hour"], inplace=True)

        categorical_cols = [
            "city",
            "weather_verdict",
            "time_of_day",
            "season_region",
        ]

        X[categorical_cols] = (
            X[categorical_cols].astype("category")
        )

        logger.info("Feature engineering completed successfully.")

        return X, Y

    except Exception:
        logger.exception("Error during feature engineering.")
        raise