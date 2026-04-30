import time
import pendulum
from airflow.sdk import DAG, task
from isodate import parse_duration


with DAG(
    dag_id="duration",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    params={
        "duration": "PT10S",  # default value
    },
) as dag:

    @task
    def use_duration(params=None):
        duration_str = params["duration"]

        try:
            duration = parse_duration(duration_str)
            seconds = int(duration.total_seconds())
        except Exception:
            raise ValueError(f"Invalid duration: {duration_str}")

        print(f"Duration: {duration_str}")
        print(f"Seconds: {seconds}")

        # keep small for testing
        time.sleep(min(seconds, 10))

    use_duration()