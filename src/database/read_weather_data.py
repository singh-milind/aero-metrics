import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def read(city):
    query = text("""
        SELECT
            time,
            city,
            temperature_2m,
            relative_humidity_2m,
            wind_speed_10m,
            wind_direction_10m,
            precipitation,
            surface_pressure
        FROM weather_data
        WHERE city = :city
        ORDER BY time
    """)


    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "city": city,
            }
        )

    if df.empty:
        raise ValueError(
            f"No historical data found for city={city}, "
        )

    df["time"] = pd.to_datetime(df["time"], utc=True)

    return df