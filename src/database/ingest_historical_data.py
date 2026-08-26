import pandas as pd
from sqlalchemy import text

from src.database.connection import engine




def ingest(df):


    # Make sure timestamp is parsed
    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    # Keep only columns that belong in historical_data
    df = df[
        [
            "time",
            "city",
            "pm2_5",
            "pm10",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
        ]
    ]

    # Remove rows that cannot identify a historical observation
    df = df.dropna(
        subset=["time", "city"]
    )

    # Remove duplicate observations inside CSV
    df = df.drop_duplicates(
        subset=["city", "time"]
    )

    query = text("""
        INSERT INTO historical_data (
            time,
            city,
            pm2_5,
            pm10,
            temperature_2m,
            relative_humidity_2m,
            wind_speed_10m,
            wind_direction_10m,
            precipitation,
            surface_pressure
        )
        VALUES (
            :time,
            :city,
            :pm2_5,
            :pm10,
            :temperature_2m,
            :relative_humidity_2m,
            :wind_speed_10m,
            :wind_direction_10m,
            :precipitation,
            :surface_pressure
        )
        ON CONFLICT (city, time)
        DO UPDATE SET
            pm2_5 = EXCLUDED.pm2_5,
            pm10 = EXCLUDED.pm10,
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

    print(f"Inserted/updated {len(records)} rows.")


if __name__ == "__main__":
    ingest()