from asyncio.log import logger

def get_weather_verdict(row):
    try:
        temp = row["temperature_2m"]
        humidity = row["relative_humidity_2m"]
        wind = row["wind_speed_10m"]
        rain = row["precipitation"]

        if rain > 2.0:
            return "Rainy / Washed"

        elif wind > 15.0:
            return "Windy & Clear"

        elif temp < 18.0 and wind < 5.0:
            return "Cold & Stagnant"

        elif temp > 35.0 and humidity < 40.0:
            return "Hot & Dry"

        elif temp > 28.0 and humidity > 70.0:
            return "Hot & Humid"

        return "Pleasant / Normal"

    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise

    except Exception as e:
        logger.exception(f"Failed to determine weather verdict: {e}")
        raise


def apply_weather_verdict(df):
    logger.info("Generating weather verdict feature...")

    try:
        df["weather_verdict"] = df.apply(get_weather_verdict, axis=1)
        logger.info("Weather verdict feature created successfully.")
    except Exception:
        logger.exception("Failed to generate weather verdict feature.")
        raise
    return df