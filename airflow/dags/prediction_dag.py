
import datetime
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    dag_id='backorder_prediction',
    schedule='0 0 * * *',
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    description='Backorder prediction pipeline dag',
    dagrun_timeout=datetime.timedelta(minutes=15),
    tags=['backorder', 'prediction']
)as dag:


    from backorder.pipeline.prediciton_pipeline import PredictionPipeline

    def start_batch_prediction(**kwargs):

        pred_pipe = PredictionPipeline()
        pred_pipe.start_batch_prediction()

    batch_prediction = PythonOperator(
        task_id='batch_prediction',
        python_callable=start_batch_prediction
    )

    start_batch_prediction