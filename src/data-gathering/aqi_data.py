import numpy as np
import pandas as pd
import requests
import time
from cities import city_coords



#final_api
def fetch_aqi_data(city_name, lat, lon, start_date, end_date):
    print(f"Fetching 5-year AQI data for {city_name}...")

    # The Open-Meteo Air Quality Endpoint
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # Setting up the parameters for the API request
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5"], # Requesting both major pollutants
        "timezone": "auto"
    }

    # 1. Make the request and convert the response to JSON
    response = requests.get(url, params=params)

    # Check if the API call was successful (Status Code 200)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return None

    data = response.json()

    # 2. Load the 'hourly' section of the JSON into a Pandas DataFrame
    df = pd.DataFrame(data['hourly'])

    # 3. Convert the 'time' text into actual Datetime objects
    df['time'] = pd.to_datetime(df['time'])

    # 4. Set 'time' as the index (This unlocks Pandas time-series powers)
    df.set_index('time', inplace=True)

    # 5. Resample into 4 parts of the day
    # '6h' tells Pandas to take the average of every 6-hour block
    # 00:00 (Night), 06:00 (Morning), 12:00 (Afternoon), 18:00 (Evening)
    df_4x_daily = df.resample('6h').mean().reset_index()

    # 6. Add the City name so we can identify it later when we merge all 28 cities
    df_4x_daily.insert(1, 'city', city_name)

    # Drop any rows where the sensor data might have been completely missing
    df_4x_daily.dropna(inplace=True)

    print(f"Success! Extracted {len(df_4x_daily)} rows for {city_name}.")
    return df_4x_daily

#final_fetching
all_cities_data = []

for capital, info in city_coords.items():
    # Fetch 5 years of data for the specific city
    df_city = fetch_aqi_data(
        city_name=capital,
        lat=info['lat'],
        lon=info['lon'],
        start_date="2021-01-01",
        end_date="2026-06-30"
    )

    if df_city is not None:
        all_cities_data.append(df_city)
    print("Pausing for 5 seconds to respect API limits...")
    time.sleep(5)

# Combine all 28 DataFrames into one massive Master Dataset
india_aqi_df = pd.concat(all_cities_data, ignore_index=True)

