from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.data_gathering.weather_data import main as weather_main
from src.data_gathering.aqi_data import main as aqi_main


if __name__ == "__main__":
    end_date = datetime.now().date()
    start_date = end_date - relativedelta(years=3)

    weather_main(
        start_date=start_date,
        end_date=end_date,
    )

    aqi_main(
        start_date=start_date,
        end_date=end_date,
    )
