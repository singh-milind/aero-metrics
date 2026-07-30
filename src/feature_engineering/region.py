
from src.feature_engineering.city_to_region import CITY_TO_REGION

def region_map(df,logger):
    logger.info("Mapping cities to regions...")

    try:
        df["region"] = df["city"].map(CITY_TO_REGION)
        if df["region"].isna().any():
            missing = df.loc[df["region"].isna(), "city"].unique()
            logger.error(f"Missing region mapping for: {missing}")
            raise ValueError("Region mapping incomplete.")
        logger.info("City to region mapping completed successfully.")
    except Exception:
        logger.exception("Failed to map cities to regions.")
        raise
    return df

