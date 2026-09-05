from sqlalchemy import text
import pandas as pd
def fetch_filtered_data(filters, engine):

    query = """
        SELECT
            time,
            city,
            pm10,
            pm2_5,
            aqi,
            temperature_2m,
            relative_humidity_2m,
            wind_speed_10m,
            wind_direction_10m,
            precipitation,
            surface_pressure,
            month,
            hour,
            day_of_week,
            time_of_day,
            is_weekend,
            weather_verdict,
            region,
            regional_season
        FROM historical_data
        WHERE 1=1
    """

    params = {}

    if filters.city is not None:
        query += " AND city = :city"
        params["city"] = filters.city

    if filters.region is not None:
        query += " AND region = :region"
        params["region"] = filters.region

    if filters.year is not None:
        query += " AND EXTRACT(YEAR FROM time) = :year"
        params["year"] = filters.year

    if filters.season is not None:
        query += " AND regional_season = :season"
        params["season"] = filters.season

    if filters.time_of_day is not None:
        query += " AND time_of_day = :time_of_day"
        params["time_of_day"] = filters.time_of_day

    if filters.is_weekend is not None:
        query += " AND is_weekend = :is_weekend"
        params["is_weekend"] = filters.is_weekend

    if filters.weather_verdict is not None:
        query += " AND weather_verdict = :weather_verdict"
        params["weather_verdict"] = filters.weather_verdict

    if filters.start_date is not None:
        query += " AND time >= :start_date"
        params["start_date"] = filters.start_date

    if filters.end_date is not None:
        query += " AND time <= :end_date"
        params["end_date"] = filters.end_date

    query += " ORDER BY time"

    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params=params
        )

    return df