from datetime import datetime
import os
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# 1️⃣ Task: Fetch weather data from Visual Crossing
def fetch_weather_data(ti):
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    location = "Senegal"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?unitGroup=metric&key={api_key}&contentType=json"

    response = requests.get(url)
    data = response.json()
    
    # Extract current temperature (Celsius)
    temp_c = data["currentConditions"]["temp"]
    print(f"Fetched temperature for {location}: {temp_c}°C")

    # Push temperature to XCom
    ti.xcom_push(key="temp_celsius", value=temp_c)


# 2️⃣ Task: Convert Celsius → Fahrenheit & Kelvin
def process_temperature(ti):
    temp_c = ti.xcom_pull(task_ids="fetch_weather", key="temp_celsius")
    
    temp_f = (temp_c * 9/5) + 32
    temp_k = temp_c + 273.15

    print(f"Converted: {temp_c}°C = {temp_f}°F = {temp_k}K")

    ti.xcom_push(key="temp_fahrenheit", value=temp_f)
    ti.xcom_push(key="temp_kelvin", value=temp_k)


# 3️⃣ Task: Save processed data to a local file
def save_to_file(ti):
    temp_c = ti.xcom_pull(task_ids="fetch_weather", key="temp_celsius")
    temp_f = ti.xcom_pull(task_ids="process_temp", key="temp_fahrenheit")
    temp_k = ti.xcom_pull(task_ids="process_temp", key="temp_kelvin")

    file_path = "/tmp/weather_data.txt"
    with open(file_path, "w") as f:
        f.write(f"Temperature data (London):\n")
        f.write(f"Celsius: {temp_c}°C\nFahrenheit: {temp_f}°F\nKelvin: {temp_k}K\n")

    print(f"Weather data saved to {file_path}")


# 4️⃣ DAG Definition
with DAG(
    dag_id="weather_xcom_dag",
    start_date=datetime(2025, 10, 7),
    schedule=None,
    catchup=False,
    tags=["xcom", "weather", "api"],
) as dag:

    fetch_weather = PythonOperator(
        task_id="fetch_weather",
        python_callable=fetch_weather_data,
    )

    process_temp = PythonOperator(
        task_id="process_temp",
        python_callable=process_temperature,
    )

    save_file = PythonOperator(
        task_id="save_file",
        python_callable=save_to_file,
    )

    # Task dependency
    fetch_weather >> process_temp >> save_file
