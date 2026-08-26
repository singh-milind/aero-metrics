import pandas as pd
from sqlalchemy import text

from src.database.connection import engine


def ingest(df):

    df = df.copy()

    # Rename DataFrame timestamp to match database column
    df = df.rename(
        columns={"time": "forecast_time"}
    )

    # Parse timestamp
    df["forecast_time"] = pd.to_datetime(
        df["forecast_time"],
        utc=True
    )

    # Keep only columns required by weather_forecasts
    df = df[
        [
            "forecast_time",
            "city",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
        ]
    ]

    # Remove invalid rows
    df = df.dropna(
        subset=["forecast_time", "city"]
    )

    # Remove duplicate city + forecast timestamp
    df = df.drop_duplicates(
        subset=["city", "forecast_time"],
        keep="last"
    )

    query = text("""
        INSERT INTO weather_forecasts (
            forecast_time,
            city,
            temperature_2m,
            relative_humidity_2m,
            wind_speed_10m,
            wind_direction_10m,
            precipitation,
            surface_pressure
        )
        VALUES (
            :forecast_time,
            :city,
            :temperature_2m,
            :relative_humidity_2m,
            :wind_speed_10m,
            :wind_direction_10m,
            :precipitation,
            :surface_pressure
        )
        ON CONFLICT (city, forecast_time)
        DO UPDATE SET
            temperature_2m = EXCLUDED.temperature_2m,
            relative_humidity_2m = EXCLUDED.relative_humidity_2m,
            wind_speed_10m = EXCLUDED.wind_speed_10m,
            wind_direction_10m = EXCLUDED.wind_direction_10m,
            precipitation = EXCLUDED.precipitation,
            surface_pressure = EXCLUDED.surface_pressure
    """)

    records = df.to_dict(
        orient="records"
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            records
        )

    print(
        f"Inserted/updated {len(records)} rows."
    )


if __name__ == "__main__":
    ingest()