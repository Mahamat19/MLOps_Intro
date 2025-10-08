from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime

# Define the DAG
with DAG(
    'simple_python_dag',
    start_date=datetime(2025, 10, 6),
    schedule='3 1 * * *',
    catchup=False,
) as dag:

    # Task 1 function
    def task_1():
        print("Task 1 is running")

    # Task 2 function
    def task_2():
        print("Task 2 is running")

    # Define tasks
    t1 = PythonOperator(
        task_id='task_1',
        python_callable=task_1
    )

    t2 = PythonOperator(
        task_id='task_2',
        python_callable=task_2
    )

    # Set task order
    t1 >> t2
