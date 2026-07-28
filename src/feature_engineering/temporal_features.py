from asyncio.log import logger
import pandas as pd

def get_time_of_day(hour):
    if 0 <= hour < 6:
        return 'Midnight'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:  # 18 to 23
        return 'Evening'

def add_time_features(df):
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_name()
    df["time_of_day"] = df["hour"].apply(get_time_of_day)
    df["is_weekend"] = (
        df["time"].dt.dayofweek.isin([5, 6]).astype(int)
    )

    return df

def get_regional_season(row):
    try:
        month = row["month"]
        region = row["region"]

        if region == "South":
            if month in [1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Southwest Monsoon"
            return "Retreating Monsoon"

        elif region == "Northeast":
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4]:
                return "Pre-Monsoon"
            elif month in [5, 6, 7, 8, 9, 10]:
                return "Extended Monsoon"
            return "Post-Monsoon"

        elif region in ["North", "Central", "East", "West", "NCR"]:
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5, 6]:
                return "Summer"
            elif month in [7, 8, 9]:
                return "Southwest Monsoon"
            return "Post-Monsoon"

        elif region == "UT":
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4]:
                return "Spring"
            elif month in [5, 6, 7, 8]:
                return "Summer"
            return "Autumn"

        return "Unknown"

    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise

    except Exception as e:
        logger.exception(f"Failed to determine season: {e}")
        raise


def apply_regional_season(df):
    logger.info("Generating regional season feature...")

    try:
        df["regional_season"] = df.apply(get_regional_season, axis=1)
        logger.info("Regional season feature created successfully.")
    except Exception:
        logger.exception("Failed to generate regional season feature.")
        raise
    return df