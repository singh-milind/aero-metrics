import numpy as np
import pandas as pd

def calculate_sub_index(C, breakpoints):
    """
    Calculate pollutant sub-index using linear interpolation.

    C = pollutant concentration
    breakpoints = [(BLO, BHI, ILO, IHI), ...]
    """

    if pd.isna(C):
        return np.nan

    for BLO, BHI, ILO, IHI in breakpoints:
        if BLO <= C <= BHI:
            return (
                ((IHI - ILO) / (BHI - BLO))
                * (C - BLO)
                + ILO
            )

    return np.nan


# PM2.5 breakpoints (µg/m³)
pm25_bp = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 1000, 400, 500),
]


# PM10 breakpoints (µg/m³)
pm10_bp = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 250, 100, 200),
    (250, 350, 200, 300),
    (350, 430, 300, 400),
    (430, 1000, 400, 500),
]


def calculate_aqi(row):
    pm25_index = calculate_sub_index(
        row["pm2_5"],
        pm25_bp
    )

    pm10_index = calculate_sub_index(
        row["pm10"],
        pm10_bp
    )

    return np.nanmax([
        pm25_index,
        pm10_index
    ])