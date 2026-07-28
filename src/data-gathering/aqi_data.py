import logging
import time
import requests
import pandas as pd
from cities import city_coords
from pathlib import Path

import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "aqi_fetch.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



def fetch_aqi_data(city_name, lat, lon, start_date, end_date):
    logger.info(f"Fetching AQI data for {city_name}")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            logger.warning(f"No hourly data for {city_name}")
            return None

        df = pd.DataFrame(data["hourly"])

        if df.empty:
            logger.warning(f"Empty data for {city_name}")
            return None

        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        df = df.resample("6h").mean().reset_index()
        df.insert(1, "city", city_name)
        df.dropna(inplace=True)

        logger.info(f"{city_name}: {len(df)} rows")

        return df

    except requests.exceptions.Timeout:
        logger.error(f"Timeout: {city_name}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error ({city_name}): {e}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed ({city_name}): {e}")

    except Exception:
        logger.exception(f"Unexpected error ({city_name})")

    return None


start_date = params["data-gathering"]["start-date"]
end_date = params["data-gathering"]["end-date"]
all_cities_data = []

for i, (city, info) in enumerate(city_coords.items(), start=1):
    try:
        logger.info(f"[{i}/{len(city_coords)}] Fetching {city}")

        df_city = fetch_aqi_data(
            city_name=city,
            lat=info["lat"],
            lon=info["lon"],
            start_date=start_date,
            end_date=end_date
        )

        if df_city is not None:
            all_cities_data.append(df_city)

    except Exception:
        logger.exception(f"Failed processing {city}")

    logger.info(f"Sleeping 5s before next city...")
    time.sleep(5)

if all_cities_data:
    india_aqi_df = pd.concat(all_cities_data, ignore_index=True)
    logger.info(f"Final dataset shape: {india_aqi_df.shape}")
else:
    logger.error("No data fetched.")
    india_aqi_df = pd.DataFrame()




DATA_DIR = ROOT_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

india_aqi_df.to_csv(DATA_DIR / "india_aqi.csv", index=False)
logger.info("Dataset saved to data/raw/india_aqi.csv")