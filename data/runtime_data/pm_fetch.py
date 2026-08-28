
import time
import requests
import pandas as pd
from src.city_info import city_info
from pathlib import Path
from src.utils.logger import get_logger
from src.database.ingest_pm_data import ingest

logger = get_logger("runtime_data_fetch")

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


def get_aqi_data_for_all_cities(start_date, end_date):
    all_cities_data = []

    for i, (city, info) in enumerate(city_info.items(), start=1):
        try:
            logger.info(f"[{i}/{len(city_info)}] Fetching {city}")

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

        logger.info(f"Sleeping 2s before next city...")
        time.sleep(2)

    if all_cities_data:
        india_aqi_df = pd.concat(all_cities_data, ignore_index=True)
        logger.info(f"Final dataset shape: {india_aqi_df.shape}")
    else:
        logger.error("No data fetched.")
        india_aqi_df = pd.DataFrame()

    return india_aqi_df 

def main():
    start_date = "2026-08-25"
    end_date = "2026-08-27"

    aqi_data = get_aqi_data_for_all_cities(start_date, end_date)

    if not aqi_data.empty:
        ingest(aqi_data)
        logger.info("Data ingestion completed successfully.")
    else:
        logger.error("No data to ingest.")
        
if __name__ == "__main__":
    main()