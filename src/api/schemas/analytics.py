from pydantic import BaseModel, field_validator, model_validator
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from src.city_info import city_info

IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = Path(__file__).resolve().parents[3]

json_path = BASE_DIR / "data" / "processed" / "metadata.json"

with open(json_path, "r") as f:
    meta_data = json.load(f)


cities = set(city_info.keys())


class AnalyticsFilter(BaseModel):

    city: str | None = None
    region: str | None = None
    year: int | None = None
    season: str | None = None
    time_of_day: str | None = None
    is_weekend: bool | None = None
    weather_verdict: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


    @field_validator("city")
    @classmethod
    def validate_city(cls, value):

        if value is None:
            return value

        value = value.strip().title()

        if value not in cities:
            raise ValueError(
                f"City '{value}' is not in the list of valid cities."
            )

        return value


    @field_validator("weather_verdict")
    @classmethod
    def validate_weather_verdict(cls, value):

        if value is None:
            return value

        allowed_verdicts = [
            "Pleasant / Normal",
            "Hot & Humid",
            "Windy & Clear",
            "Rainy / Washed",
            "Cold & Stagnant",
            "Hot & Dry",
        ]

        if value not in allowed_verdicts:
            raise ValueError(
                f"Weather verdict '{value}' is not valid."
            )

        return value


    @model_validator(mode="after")
    def validate_region(self):

        if self.city is None or self.region is None:
            return self

        expected_region = city_info[self.city]["region"]

        if expected_region != self.region:
            raise ValueError(
                f"Region '{self.region}' does not match "
                f"the region of city '{self.city}'."
            )

        return self


    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, value, info):

        if value is None:
            return value

        min_date = datetime.fromisoformat(
            meta_data["date_range"]["start"]
        ).replace(tzinfo=IST)

        max_date = datetime.fromisoformat(
            meta_data["date_range"]["end"]
        ).replace(tzinfo=IST)

        if info.field_name == "start_date" and value < min_date:
            raise ValueError(
                f"Start date '{value}' is earlier than "
                f"the minimum allowed date '{min_date}'."
            )

        if info.field_name == "end_date" and value > max_date:
            raise ValueError(
                f"End date '{value}' is later than "
                f"the maximum allowed date '{max_date}'."
            )

        return value