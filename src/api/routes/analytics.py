
from src.api.services.analytics import fetch_filtered_data
from src.database.connection import engine
from fastapi import APIRouter
from src.api.schemas.analytics import AnalyticsFilter
router = APIRouter()

import numpy as np

@router.post("/trends")
def get_analytics( y_axis: str , x_axis: str, filters: AnalyticsFilter):
    df = fetch_filtered_data(filters, engine)

    # Pandas analytics
    result = (
        df.groupby(x_axis)[y_axis]
        .mean()
        .reset_index()
    )

    return result.to_dict(orient="records")
