import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

def ingest(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "city", "pm2_5", "pm10"]].dropna(subset=["time", "city"])
    df = df.drop_duplicates(subset=["city", "time"])

    delete_query = text("DELETE FROM pm_data")

    insert_query = text("""
        INSERT INTO pm_data (time, city, pm2_5, pm10)
        VALUES (:time, :city, :pm2_5, :pm10)
    """)

    records = df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(delete_query)
        connection.execute(insert_query, records)

    print(f"Deleted old PM data and inserted {len(records)} fresh PM rows.")