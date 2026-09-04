import pendulum

from airflow.sdk import dag, task
from airflow.timetables.trigger import CronTriggerTimetable

from data.runtime_data.pm_fetch import main as fetch_pm_data
from data.runtime_data.weather_fetch import main as fetch_weather_data


@dag(
    dag_id="runtime_data_fetch",
    start_date=pendulum.datetime(
        2023, 6, 30,
        tz="Asia/Kolkata",
    ),
    schedule=CronTriggerTimetable(
        cron="30 6,18 * * *",
        timezone="Asia/Kolkata",
    ),
    catchup=False,
)
def runtime_data_fetch_dag():

    @task.python
    def pm_fetch():
        fetch_pm_data()

    @task.python
    def weather_fetch():
        fetch_weather_data()

    pm_fetch()
    weather_fetch()


runtime_data_fetch_dag()