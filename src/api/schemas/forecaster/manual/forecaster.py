from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from src.utils.logger import get_logger
from src.city_info import city_info
from src.api.schemas.forecaster.manual.forecaster_features import compute_features
from src.api.schemas.forecaster.manual.pm_features import add_pm_features
from src.api.schemas.forecaster.manual.ratio_features import add_ratio_features
from src.database.read_pm_data import read as read_pm
from src.database.read_weather_data import read as read_weather
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

logger = get_logger("forecaster_api")
cities = set(city_info.keys())

class ForecasterInput(BaseModel):
    target_time: Annotated[datetime, Field(...)]
    city: Annotated[str, Field(...)]
    temperature_2m: Annotated[float, Field(...,gt=-11,lt=50, description="Temperature in °C")]
    relative_humidity_2m: Annotated[float, Field(...,gt=-1,lt=100, description="Relative humidity in %")]
    wind_speed_10m: Annotated[float, Field(...,gt=-1,lt=20, description="Wind speed in m/s")]
    wind_direction_10m: Annotated[float, Field(...,ge=-1,lt=360, description="Wind direction in degrees")]
    surface_pressure: Annotated[float, Field(...,gt=900,lt=1050, description="Atmospheric pressure in hPa")]
    precipitation: Annotated[float, Field(...,gt=-1,le=15, description="Precipitation in mm")]
    pm_2_5_lag_12h: Annotated[float, Field(...,gt=-1, description="PM2.5 concentration 12 hours ago in µg/m³")]
    pm_10_lag_12h: Annotated[float, Field(...,gt=-1, description="PM10 concentration 12 hours ago in µg/m³")]
    pm_2_5_lag_24h: Annotated[float, Field(...,gt=-1, description="PM2.5 concentration 24 hours ago in µg/m³")]
    pm_10_lag_24h: Annotated[float, Field(...,gt=-1, description="PM10 concentration 24 hours ago in µg/m³")]
    pm_2_5_lag_48h: Annotated[float, Field(...,gt=-1, description="PM2.5 concentration 48 hours ago in µg/m³")]
    pm_10_lag_48h: Annotated[float, Field(...,gt=-1, description="PM10 concentration 48 hours ago in µg/m³")]

    @field_validator("city")
    @classmethod
    def validate_city(cls, value):
        value = value.strip().title()
        if value not in cities:
            raise ValueError(
                f"City '{value}' is not in the list of valid cities."
            )
        return value

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
            microsecond=0,
        ).replace(tzinfo=None)

        now_time = now.replace(
            hour=(now.hour // 6) * 6,
            minute=0,
            second=0,
            microsecond=0,
        ).replace(tzinfo=None)

        if not now_time <= target_time <= now_time + timedelta(days=2):
            raise ValueError(
                "Forecast target must be between now and 2 days from now."
            )

        self.target_time = target_time

        return self


def prepare_input_pm25(data: ForecasterInput, target_time: datetime, now_time: datetime):
    target_time = pd.Timestamp(target_time)
    now_time = pd.Timestamp(now_time)
    given_data = pd.DataFrame([data.model_dump()])
    df_pm = read_pm(city=data.city)
    df_pm = add_pm_features(df_pm, given_data)
    df_pm["time"] = pd.to_datetime(df_pm["time"])
    
    df_pm = df_pm[df_pm["time"] == now_time].copy()
    df_weather = read_weather(city=data.city)
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df_weather = compute_features(df_weather, logger, target_time=target_time, given_data=given_data)
    df_pm.drop(columns=["time"], inplace=True)
    df_weather.drop(columns=["time"], inplace=True)
    return pd.merge(df_pm, df_weather, on=["city"], how="inner")

def prepare_input_pm10(data: ForecasterInput, target_time: datetime,now_time: datetime):
    target_time = pd.Timestamp(target_time)
    now_time = pd.Timestamp(now_time)
    df_pm = read_pm(city=data.city)
    given_data = pd.DataFrame([data.model_dump()])
    df_pm = add_ratio_features(df_pm, given_data)
    df_pm["time"] = pd.to_datetime(df_pm["time"])
    
    df_pm = df_pm[df_pm["time"] == now_time].copy()
    df_weather = read_weather(city=data.city)
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df_weather = compute_features(df_weather, logger, target_time=target_time, given_data=given_data)
    df_pm.drop(columns=["time"], inplace=True)
    df_weather.drop(columns=["time"], inplace=True)
    return pd.merge(df_pm, df_weather, on=["city"], how="inner")