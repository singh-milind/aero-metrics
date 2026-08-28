import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def clear_weather_data():
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM weather_data"))
    print("Deleted all existing weather data.")

def ingest(df, data_type):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "city", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation", "surface_pressure"]].dropna(subset=["time", "city"])
    df = df.drop_duplicates(subset=["city", "time"])
    df["data_type"] = data_type
    query = text("""
        INSERT INTO weather_data (
            time, city, data_type,
            temperature_2m, relative_humidity_2m,
            wind_speed_10m, wind_direction_10m,
            precipitation, surface_pressure
        )
        VALUES (
            :time, :city, :data_type,
            :temperature_2m, :relative_humidity_2m,
            :wind_speed_10m, :wind_direction_10m,
            :precipitation, :surface_pressure
        )
    """)
    records = df.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(query, records)
    print(f"Inserted {len(records)} {data_type} weather rows.")

