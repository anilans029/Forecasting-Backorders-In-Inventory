#!bin/sh
nohup airflow scheduler &
airflow webserver -p 9090 &
gunicorn --workers=4 --bind 0.0.0.0:5000 app:app