import time
import requests
import pandas as pd
from zoneinfo import ZoneInfo
from src.city_info import city_info
from src.utils.logger import get_logger
from src.database.ingest_weather_data import ingest, clear_weather_data

IST = ZoneInfo("Asia/Kolkata")
logger = get_logger("runtime_data_fetch")

AGGREGATION_LOGIC = {
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean",
    "wind_direction_10m": "mean",
    "precipitation": "sum",
    "surface_pressure": "mean",
}

def process_weather_dataframe(df, city_name):
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"])

    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(IST)
    else:
        df["time"] = df["time"].dt.tz_convert(IST)

    df.set_index("time", inplace=True)

    df = df.resample("6h").agg(AGGREGATION_LOGIC).reset_index()

    df["time"] = df["time"].dt.tz_localize(None)

    df.insert(1, "city", city_name)
    df.dropna(inplace=True)

    return df

def fetch_historical_data(city_name, lat, lon, start_date, end_date):
    logger.info(f"Fetching historical weather for {city_name}...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
        ],
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            logger.warning(f"No historical data for {city_name}")
            return None

        df = pd.DataFrame(data["hourly"])

        if df.empty:
            logger.warning(f"Empty historical data for {city_name}")
            return None

        df = process_weather_dataframe(df, city_name)

        if df is None or df.empty:
            logger.warning(f"No processed historical data for {city_name}")
            return None

        logger.info(f"{city_name}: {len(df)} historical rows")

        return df

    except requests.exceptions.Timeout:
        logger.error(f"Request timed out while fetching {city_name}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {city_name}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {city_name}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while processing {city_name}: {e}")

    return None

def get_historical_data(start_date, end_date):
    all_cities_data = []

    for i, (city, info) in enumerate(city_info.items(), start=1):
        logger.info(f"[{i}/{len(city_info)}] Processing historical data: {city}")

        try:
            df_city = fetch_historical_data(
                city_name=city,
                lat=info["lat"],
                lon=info["lon"],
                start_date=start_date,
                end_date=end_date,
            )

            if df_city is not None:
                all_cities_data.append(df_city)

        except Exception:
            logger.exception(f"Failed to process {city}")

        logger.info("Sleeping 2 seconds before next request...")
        time.sleep(2)

    if not all_cities_data:
        logger.error("No historical data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_cities_data, ignore_index=True)

    logger.info(f"Historical dataset shape: {df.shape}")

    return df

def fetch_forecast_data(city_name, lat, lon):
    logger.info(f"Fetching forecast weather for {city_name}...")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
        ],
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            logger.warning(f"No forecast data for {city_name}")
            return None

        df = pd.DataFrame(data["hourly"])

        if df.empty:
            logger.warning(f"Empty forecast data for {city_name}")
            return None

        df = process_weather_dataframe(df, city_name)

        if df is None or df.empty:
            logger.warning(f"No processed forecast data for {city_name}")
            return None

        logger.info(f"{city_name}: {len(df)} forecast rows")

        return df

    except requests.exceptions.Timeout:
        logger.error(f"Request timed out while fetching {city_name}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {city_name}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {city_name}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while processing {city_name}: {e}")

    return None

def get_forecast_data():
    all_cities_data = []

    for i, (city, info) in enumerate(city_info.items(), start=1):
        logger.info(f"[{i}/{len(city_info)}] Processing forecast data: {city}")

        try:
            df_city = fetch_forecast_data(
                city_name=city,
                lat=info["lat"],
                lon=info["lon"],
            )

            if df_city is not None:
                all_cities_data.append(df_city)

        except Exception:
            logger.exception(f"Failed to process {city}")

        logger.info("Sleeping 2 seconds before next request...")
        time.sleep(2)

    if not all_cities_data:
        logger.error("No forecast data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_cities_data, ignore_index=True)

    logger.info(f"Forecast dataset shape: {df.shape}")

    return df

def main():
    start_time = time.time()

    logger.info("Starting weather data ingestion...")

    now = pd.Timestamp.now(tz=IST)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    logger.info(f"Historical range: {start_date} -> {end_date}")

    historical_weather_df = get_historical_data(
        start_date=start_date,
        end_date=end_date,
    )

    forecast_weather_df = get_forecast_data()

    if historical_weather_df.empty and forecast_weather_df.empty:
        logger.error("No weather data fetched. Database will not be modified.")
        return

    clear_weather_data()

    if not historical_weather_df.empty:
        ingest(historical_weather_df, data_type="observed")
        logger.info("Historical weather data saved successfully.")
    else:
        logger.warning("Historical dataset is empty.")

    if not forecast_weather_df.empty:
        ingest(forecast_weather_df, data_type="forecast")
        logger.info("Forecast weather data saved successfully.")
    else:
        logger.warning("Forecast dataset is empty.")

    elapsed_time = time.time() - start_time

    logger.info(f"Weather ingestion completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()