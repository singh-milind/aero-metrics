import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def read(city):
    query = text("""
        SELECT
            time,
            city,
            pm2_5,
            pm10
        FROM pm_data
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