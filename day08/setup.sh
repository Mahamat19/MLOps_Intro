# source .venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
airflow users create --username Mahamat19 --firstname Mahamat --lastname Assouna --role Admin --email amahamat@aimsammi.org
# airflow db migrate
# airflow users create \
#     --username Mahamat19 \
#     --firstname Mahamat \
#     --lastname Assouna \
#     --role Admin \
#     --email amahamat@aimsammi.org \
#     --password Assouna19@

# Dqx2emhWyEG9e2bY