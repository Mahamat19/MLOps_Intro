from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime

# Define the DAG
with DAG(
    'bash_python_dag',
    start_date=datetime(2025, 10, 6),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Bash task
    bash_task = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    # Python task
    def greet():
        print("Hello from Python task!")

    python_task = PythonOperator(
        task_id='greet_task',
        python_callable=greet,
    )

    # Task order
    bash_task >> python_task
