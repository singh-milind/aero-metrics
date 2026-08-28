from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Annotated,Literal
import pandas as pd

from src.utils.logger import get_logger
from src.city_info import city_info
from src.api.schemas.predictor.predictor_features import compute_features

logger = get_logger('predictor_api')
cities = set(list(city_info.keys()))

class PredictorInput(BaseModel):
    """
    Input schema for the predictor API.
    """
    temperature_2m: Annotated[float, Field(...,gt=-11,lt=50, description="Temperature in °C")]
    relative_humidity_2m: Annotated[float, Field(...,gt=-1,lt=100, description="Relative humidity in %")]
    wind_speed_10m: Annotated[float, Field(...,gt=-1,lt=20, description="Wind speed in m/s")]
    wind_direction_10m: Annotated[float, Field(...,ge=-1,lt=360, description="Wind direction in degrees")]
    surface_pressure: Annotated[float, Field(...,gt=900,lt=1050, description="Atmospheric pressure in hPa")]
    precipitation: Annotated[float, Field(...,gt=-1,le=15, description="Precipitation in mm")]
    month: Annotated[int, Field(...,ge=1,le=12, description="Month of the year (1-12)")]
    day_of_week: Annotated[int, Field(...,ge=-1,le=6, description="Day of the week (0=Monday, 6=Sunday)")]
    time_of_day: Annotated[Literal['Morning', 'Afternoon', 'Evening', 'Midnight'], Field(..., description="Time of the day")]
    city: Annotated[str, Field(..., description="City name")]
    
    @field_validator('city')
    def validate_city(cls, value):
        value = value.strip().title()
        if value not in cities:
            raise ValueError(f"City '{value}' is not in the list of valid cities.")
        return value
    
def prepare_input(data: PredictorInput):
    df = pd.DataFrame([data.model_dump()])
    return compute_features(df, logger)


    
    
    
   