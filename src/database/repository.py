from datetime import datetime, timedelta

import pandas as pd 
from sqlalchemy import text


def get_historical_data(
    db,
    city: str,
    target_time: datetime,
) -> pd.DataFrame:

    start_time = target_time - timedelta(hours=48)

    query = text("""
        SELECT
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
        FROM historical_data
        WHERE city = :city
          AND time >= :start_time
          AND time <= :target_time
        ORDER BY time ASC
    """)

    result = db.execute(
        query,
        {
            "city": city,
            "start_time": start_time,
            "target_time": target_time,
        },
    )

    rows = result.mappings().all()

    return pd.DataFrame(rows)