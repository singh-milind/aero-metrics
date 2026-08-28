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

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df['time'] = df['time'] + pd.Timedelta(minutes=30)  # Adjusting time to the middle of the 6-hour interval

    return df