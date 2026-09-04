from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

from airflow.sdk import dag, task
from airflow.timetables.trigger import DeltaTriggerTimetable


@dag(
    dag_id="model_training",
    start_date=datetime(
        2023, 6, 30,
        tzinfo=ZoneInfo("Asia/Kolkata")
    ),
    schedule=DeltaTriggerTimetable(delta=timedelta(days=15)),
    catchup=False,
)
def model_training_dag():

    @task.bash
    def train_model():
        command = "cd /opt/aero-metrics && dvc repro"
        return command

    train_model()


model_training_dag()