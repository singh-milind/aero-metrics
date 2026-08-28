from datetime import datetime, timedelta, timezone
from typing import Annotated
from pydantic import BaseModel, Field, field_validator, model_validator
from src.utils.logger import get_logger
from src.city_info import city_info
from src.api.schemas.forecaster.auto.forecaster_features import compute_features
from src.api.schemas.forecaster.auto.pm_features import add_pm_features
from src.api.schemas.forecaster.auto.ratio_features import add_ratio_features
from src.database.read_pm_data import read as read_pm
from src.database.read_weather_data import read as read_weather
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = get_logger("forecaster_api")
cities = set(city_info.keys())

class ForecasterInput(BaseModel):
    target_time: Annotated[datetime, Field(..., description="Forecast target datetime")]
    city: Annotated[str, Field(..., description="City name")]

    @field_validator("city")
    @classmethod
    def validate_city(cls, value):
        value = value.strip().title()
        if value not in cities:
            raise ValueError(f"City '{value}' is not in the list of valid cities.")
        return value


IST = ZoneInfo("Asia/Kolkata")

@model_validator(mode="after")
def validate_target_time(self):
    now = datetime.now(IST)
    target_time = self.target_time

    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=IST)
    else:
        target_time = target_time.astimezone(IST)

    target_time = target_time.replace(
        hour=(target_time.hour // 6) * 6,
        minute=0,
        second=0,
        microsecond=0
    )

    now = now.replace(
        hour=(now.hour // 6) * 6,
        minute=0,
        second=0,
        microsecond=0
    )

    if not now <= target_time <= now + timedelta(days=2):
        raise ValueError("Forecast target must be between now and 2 days from now.")

    self.target_time = target_time
    return self

def prepare_input_pm25(data: ForecasterInput, target_time: datetime):
    target_time = pd.to_datetime(target_time)
    df_pm = read_pm(city=data.city)
    df_pm = add_pm_features(df_pm)
    df_pm["time"] = pd.to_datetime(df_pm["time"], utc=True)
    df_pm = df_pm[df_pm["time"] == df_pm["time"].max()].copy()
    df_weather = read_weather(city=data.city)
    df_weather = compute_features(df_weather, logger, target_time=target_time)
    df_pm.drop(columns=["time"], inplace=True)
    df_weather.drop(columns=["time"], inplace=True)
    df = pd.merge(df_pm, df_weather, on=["city"], how="outer")
    return df

def prepare_input_pm10(data: ForecasterInput, target_time: datetime):
    target_time = pd.to_datetime(target_time, utc=True).floor("6h")
    df_pm = read_pm(city=data.city)
    df_pm = add_ratio_features(df_pm)
    df_pm["time"] = pd.to_datetime(df_pm["time"], utc=True)
    df_pm = df_pm[df_pm["time"] == df_pm["time"].max()].copy()
    df_weather = read_weather(city=data.city)
    df_weather = compute_features(df_weather, logger, target_time=target_time)
    df_pm.drop(columns=["time"], inplace=True)
    df_weather.drop(columns=["time"], inplace=True)
    df = pd.merge(df_pm, df_weather, on=["city"], how="outer")
    return df