import subprocess
from pathlib import Path

from src.utils.blob_storage import (
    download_blob,
    upload_directory,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # ============================================================
    # 1. DOWNLOAD LATEST RAW DATA FROM AZURE BLOB
    # ============================================================

    download_blob(
        "data",
        "raw/weather_data.csv",
        str(RAW_DATA_DIR / "weather_data.csv"),
    )

    download_blob(
        "data",
        "raw/aqi_data.csv",
        str(RAW_DATA_DIR / "aqi_data.csv"),
    )

    # ============================================================
    # 2. RUN COMPLETE DVC PIPELINE
    # ============================================================

    subprocess.run(
        ["dvc", "repro"],
        cwd=ROOT_DIR,
        check=True,
    )

    # ============================================================
    # 3. UPLOAD PREDICTOR MODELS
    # ============================================================

    predictor_base = (
        ROOT_DIR
        / "src"
        / "model_building"
        / "predictor"
    )

    upload_directory(
        "models",
        str(predictor_base / "pm25_model"),
        "predictor/pm25",
    )

    upload_directory(
        "models",
        str(predictor_base / "pm10_model"),
        "predictor/pm10",
    )

    # ============================================================
    # 4. UPLOAD FORECASTER MODELS
    # ============================================================

    forecaster_base = (
        ROOT_DIR
        / "src"
        / "model_building"
        / "forecaster"
    )

    horizons = [ 
        "t_model",
        "t12_model",
        "t24_model",
        "t48_model",
    ]

    # PM2.5 forecasters
    for horizon in horizons:
        upload_directory(
            "models",
            str(
                forecaster_base
                / "pm25_model"
                / horizon
            ),
            f"forecaster/pm25/{horizon}",
        )

    # PM10 forecasters
    for horizon in horizons:
        upload_directory(
            "models",
            str(
                forecaster_base
                / "pm10_model"
                / horizon
            ),
            f"forecaster/pm10/{horizon}",
        )

    # ============================================================
    # 5. UPLOAD ALL METRICS
    # ============================================================

    upload_directory(
        "metrics",
        str(ROOT_DIR / "metrics"),
    )