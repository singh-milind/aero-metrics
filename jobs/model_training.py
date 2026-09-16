import subprocess
from pathlib import Path

from src.utils.blob_storage import download_blob_bytes


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_raw_data():
    """Download raw CSV files from Azure Blob Storage."""

    weather_data = download_blob_bytes(
        container_name="data",
        blob_name="raw/weather_data.csv",
    )

    aqi_data = download_blob_bytes(
        container_name="data",
        blob_name="raw/aqi_data.csv",
    )

    # Save downloaded bytes locally for DVC preprocessing
    (RAW_DATA_DIR / "weather_data.csv").write_bytes(
        weather_data
    )

    (RAW_DATA_DIR / "aqi_data.csv").write_bytes(
        aqi_data
    )


if __name__ == "__main__":

    # 1. DOWNLOAD RAW DATA REQUIRED BY DVC

    download_raw_data()

    # 2. RUN COMPLETE DVC PIPELINE

    subprocess.run(
        ["dvc", "repro"],
        cwd=ROOT_DIR,
        check=True,
    )