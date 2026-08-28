import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def ingest(df, data_type):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
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
        ON CONFLICT (city, time, data_type)
        DO UPDATE SET
            temperature_2m = EXCLUDED.temperature_2m,
            relative_humidity_2m = EXCLUDED.relative_humidity_2m,
            wind_speed_10m = EXCLUDED.wind_speed_10m,
            wind_direction_10m = EXCLUDED.wind_direction_10m,
            precipitation = EXCLUDED.precipitation,
            surface_pressure = EXCLUDED.surface_pressure
    """)

    records = df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)

    print(f"Inserted/updated {len(records)} {data_type} weather rows.")