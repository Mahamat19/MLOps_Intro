# Launch webserver
source .venv/bin/activate
airflow db migrate
export AIRFLOW_HOME=${PWD}/airflow
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export VISUAL_CROSSING_API_KEY=2G8Z7U4PVQKL59TR2HSA9TBW3
# airflow api-server --port 8080  # http://localhost:8080
airflow standalone