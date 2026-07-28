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
        logging.FileHandler(LOG_DIR / "weather_fetch.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#weather_api
def fetch_weather_data(city_name, lat, lon, start_date, end_date):
    print(f"Fetching Weather data for {city_name}...")

    # Open-Meteo Historical Weather Endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "wind_direction_10m", "precipitation", "surface_pressure"],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data['hourly'])

    # Convert time string to Datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # The Smart Resampling Dictionary
    aggregation_logic = {
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'wind_speed_10m': 'mean',
        'wind_direction_10m': 'mean',
        'precipitation': 'sum',
        'surface_pressure': 'mean'
    }

    # Group into 4 parts of the day (6-hour blocks) using the logic above
    df_4x_daily = df.resample('6h').agg(aggregation_logic).reset_index()

    # Add the City name
    df_4x_daily.insert(1, 'city', city_name)
    df_4x_daily.dropna(inplace=True)

    print(f"Success! Extracted {len(df_4x_daily)} weather rows for {city_name}.")
    return df_4x_daily


start_date = params["data-gathering"]["start-date"]
end_date = params["data-gathering"]["end-date"]
all_cities_data = []

for capital, info in city_coords.items():
    # Fetch 5 years of data for the specific city
    df_city = fetch_weather_data(
        city_name=capital,
        lat=info['lat'],
        lon=info['lon'],
        start_date=start_date,
        end_date=end_date
    )

    if df_city is not None:
        all_cities_data.append(df_city)
    print("Pausing for 5 seconds to respect API limits...")
    time.sleep(5)

# Combine all DataFrames into one massive Master Dataset
master_india_weather_df = pd.concat(all_cities_data, ignore_index=True)