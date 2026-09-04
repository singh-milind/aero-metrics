
import time
import requests
import pandas as pd
from src.city_info import city_info
from pathlib import Path
from src.utils.logger import get_logger

ROOT_DIR = Path(__file__).resolve().parents[2]
logger = get_logger("weather_fetch")

def fetch_weather_data(city_name, lat, lon, start_date, end_date):
    logger.info(f"Fetching weather data for {city_name}...")

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
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data["hourly"])

        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        aggregation_logic = {
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "mean",
            "wind_direction_10m": "mean",
            "precipitation": "sum",
            "surface_pressure": "mean",
        }

        df_4x_daily = (
            df.resample("6h")
            .agg(aggregation_logic)
            .reset_index()
        )

        df_4x_daily.insert(1, "city", city_name)
        df_4x_daily.dropna(inplace=True)

        logger.info(
            f"Successfully extracted {len(df_4x_daily)} weather records for {city_name}"
        )

        return df_4x_daily

    except requests.exceptions.Timeout:
        logger.error(f"Request timed out while fetching {city_name}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {city_name}: {e}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {city_name}: {e}")

    except Exception as e:
        logger.exception(f"Unexpected error while processing {city_name}: {e}")

    return None

def main(start_date, end_date):
    all_cities_data = []

    for i, (city, info) in enumerate(city_info.items(), start=1):
        logger.info(f"[{i}/{len(city_info)}] Processing {city}")

        try:
            df_city = fetch_weather_data(
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

    if all_cities_data:
        india_weather_df = pd.concat(all_cities_data, ignore_index=True)
        logger.info(f"Final dataset shape: {india_weather_df.shape}")
    else:
        logger.error("No data fetched.")
        india_weather_df = pd.DataFrame()


    DATA_DIR = ROOT_DIR / "data" / "raw"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    india_weather_df.to_csv(DATA_DIR / "weather_data.csv", index=False)
    logger.info("Dataset saved to data/raw/weather_data.csv")