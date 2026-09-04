from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

from airflow.sdk import dag, task
from airflow.timetables.trigger import DeltaTriggerTimetable

from src.data_gathering.weather_data import main as weather_main
from src.data_gathering.aqi_data import main as aqi_main


@dag(
    dag_id="data_gathering",
    start_date=datetime(
        2023,
        6,
        30,
        tzinfo=ZoneInfo("Asia/Kolkata"),
    ),
    schedule=DeltaTriggerTimetable(
        delta=timedelta(days=15)
    ),
    catchup=False,
)
def data_gathering_dag():

    @task.python
    def collect_weather_data():
        end_date = datetime.now().date()
        start_date = end_date - relativedelta(years=3)

        weather_main(
            start_date=start_date,
            end_date=end_date,
        )

    @task.python
    def collect_aqi_data():
        end_date = datetime.now().date()
        start_date = end_date - relativedelta(years=3)

        aqi_main(
            start_date=start_date,
            end_date=end_date,
        )

    collect_weather_data()
    collect_aqi_data()


data_gathering_dag()