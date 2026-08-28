import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def ingest(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[["time", "city", "pm2_5", "pm10"]].dropna(subset=["time", "city"])
    df = df.drop_duplicates(subset=["city", "time"])

    query = text("""
        INSERT INTO pm_data (time, city, pm2_5, pm10)
        VALUES (:time, :city, :pm2_5, :pm10)
        ON CONFLICT (city, time)
        DO UPDATE SET
            pm2_5 = EXCLUDED.pm2_5,
            pm10 = EXCLUDED.pm10
    """)

    records = df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)

    print(f"Inserted/updated {len(records)} PM rows.")