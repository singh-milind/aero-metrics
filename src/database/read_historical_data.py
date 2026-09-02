import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def read():
    query = text("""
        SELECT * FROM historical_data
    """)


    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
        )

    if df.empty:
        raise ValueError(
            f"No historical data found, "
        )

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df['time'] = df['time'] + pd.Timedelta(minutes=30)  # Adjusting time to the middle of the 6-hour interval

    return df
