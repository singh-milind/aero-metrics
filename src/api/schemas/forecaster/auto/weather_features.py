import pandas as pd
import numpy as np

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)
    
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
    df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_precip"] = df["wind_speed_10m"] * df["precipitation"]
    df["pressure_temp"] = df["surface_pressure"] * df["temperature_2m"]

    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))
    df.drop(columns=["wind_direction_10m"], inplace=True)
        
    df["stagnation_index"] = (df["relative_humidity_2m"] / (df["wind_speed_10m"] + 1))
        
    df["dry_stagnation"] = (df["relative_humidity_2m"] * (1 / (df["wind_speed_10m"] + 1)))
        
        
    df["no_rain"] = (df["precipitation"] == 0).astype(int)

    df["weather_verdict"] = df.apply(get_weather, axis=1)

    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "surface_pressure",
        "precipitation",
    ]

    for col in weather_cols:
        df[f"{col}_change_6h"] = df[col].diff(1)
        df[f"{col}_change_12h"] = df[col].diff(2)
        df[f"{col}_change_24h"] = df[col].diff(4)
        df[f"{col}_change_48h"] = df[col].diff(8)
    return df