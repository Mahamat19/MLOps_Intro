from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator

# 1️ Define Python function that pushes data to XCom
def push_timestamp(ti):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ti.xcom_push(key='current_time', value=timestamp)
    print(f"Pushed timestamp: {timestamp}")

# 2️ Define the DAG
with DAG(
    'xcom_example_dag',
    start_date=datetime(2025, 10, 7),
    schedule=None,   # Run manually or via UI
    catchup=False,
    tags=['xcom', 'example']
) as dag:

    # 3️ Python task pushes a value
    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_timestamp
    )

    # 4️ Bash task pulls the value and prints it
    pull_task = BashOperator(
        task_id='pull_task',
        bash_command='echo "Pulled timestamp: {{ ti.xcom_pull(task_ids=\'push_task\', key=\'current_time\') }}"'
    )

    # 5️ Define task order
    push_task >> pull_task
