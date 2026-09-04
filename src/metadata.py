import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger


ROOT_DIR = Path(__file__).resolve().parents[1]
logger = get_logger("metadata_creation")


def load_data(logger):
    PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "engineered_features.csv")
        logger.info("Engineered features loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Missing input file: {e.filename}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load datasets: {e}")
        raise
    return df

def create_metadata():
    df = load_data(logger)
    df["time"]= pd.to_datetime(df["time"])
    metadata = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns},
        "min_values": {col: df[col].min() for col in df.select_dtypes(include=[np.number]).columns},
        "max_values": {col: df[col].max() for col in df.select_dtypes(include=[np.number]).columns},
        "mean_values": {col: df[col].mean() for col in df.select_dtypes(include=[np.number]).columns},
        "std_values": {col: df[col].std() for col in df.select_dtypes(include=[np.number]).columns},
        "date_range": {
            "start": df["time"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": df["time"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    metadata_path = ROOT_DIR / "data" / "processed" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metadata).to_json(metadata_path, indent=4)
    logger.info(f"Metadata created and saved to {metadata_path}")
    
def main():
    create_metadata()
    
if __name__ == "__main__":
    main()