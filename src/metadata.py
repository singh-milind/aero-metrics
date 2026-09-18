import json
import pandas as pd
import numpy as np

from pathlib import Path

from src.utils.logger import get_logger
from src.utils.blob_storage import (
    upload_metadata,
    load_csv,
    numpy_json_serializer,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

logger = get_logger("metadata_creation")

# 1. LOAD DATA

def load_data(logger):

    processed_data_dir = (
        ROOT_DIR / "data" / "processed"
    )

    processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        df = load_csv(
            container_name="data",
            blob_name="processed/engineered_features.csv",
        )

        logger.info(
            "Engineered features loaded successfully."
        )

        return df

    except FileNotFoundError as e:

        logger.error(
            f"Missing input file: {e.filename}"
        )

        raise

    except Exception as e:

        logger.exception(
            f"Failed to load datasets: {e}"
        )

        raise


# 2. CREATE METADATA

def create_metadata():

    df = load_data(logger)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    metadata = {

        "columns": df.columns.tolist(),

        "dtypes": (
            df.dtypes
            .apply(lambda x: x.name)
            .to_dict()
        ),

        "num_rows": len(df),

        "num_columns": len(df.columns),

        "missing_values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "unique_values": {
            col: df[col].nunique()
            for col in df.columns
        },

        "min_values": {
            col: df[col].min()
            for col in numeric_columns
        },

        "max_values": {
            col: df[col].max()
            for col in numeric_columns
        },

        "mean_values": {
            col: df[col].mean()
            for col in numeric_columns
        },

        "std_values": {
            col: df[col].std()
            for col in numeric_columns
        },

        "date_range": {

            "start": (
                df["time"]
                .min()
                .strftime("%Y-%m-%d %H:%M:%S")
            ),

            "end": (
                df["time"]
                .max()
                .strftime("%Y-%m-%d %H:%M:%S")
            ),

        },

    }

    # 3. SAVE METADATA LOCALLY

    metadata_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "metadata.json"
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=numpy_json_serializer,
        )

    logger.info(
        f"Metadata saved locally: {metadata_path}"
    )

    # 4. UPLOAD METADATA TO AZURE

    upload_metadata(
        container_name="data",
        blob_name="processed/metadata.json",
        metadata=metadata,
        logger=logger,
        artifact_name="metadata",
    )

    logger.info(
        "Metadata created and uploaded successfully."
    )


# 5. MAIN

def main():
    create_metadata()


if __name__ == "__main__":
    main()