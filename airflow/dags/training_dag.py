from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

with DAG(
            dag_id="backorder_train",
            tags=["bc_train"],
            default_args={'retries': 2},
            # [END default_args]
            description='Machine Learning Backorder prediction Project',
            schedule_interval="@weekly",
            start_date=pendulum.datetime(2022, 12, 19, tz="UTC"),
            catchup=False
)as dag:


    from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
    from backorder.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
    from backorder.entity.config_entity import ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
    from backorder.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
    from backorder.entity.artifact_entity import ModelTrainerArtiact, ModelEvaluationArtifact, ModelPusherArtifact
    from backorder.components import DataIngestion, DataValidation, DataTransformation, ModelTrainer, ModelEvaluation,ModelPusher
    from backorder.utils import save_artifacts_to_s3_and_clear_local, update_recent_batch_in_meta_data

    training_config=  TrainingConfigurationManager()

    def ingesting_data(**kwargs) -> DataIngestionArtifact:
        ti = kwargs["ti"]
        data_ingestion_config = training_config.get_dataingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        ti.xcom_push("data_ingestion_artifact",data_ingestion_artifact)


    def validating_data(**kwargs) -> DataValidationArtifact:
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(task_ids = "data_ingestion", key= "data_ingestion_artifact")
        # data_ingestion_artifact = DataIngestionArtifact(data_ingestion_artifact)
        print("***********",data_ingestion_artifact)
        data_validation_config = training_config.get_data_validatin_config()
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                            data_validation_config=data_validation_config)

        data_validation_artifact = data_validation.initiate_data_validation()
        ti.xcom_push("data_validation_artifact",data_validation_artifact)

    def transforming_data(**kwargs) -> DataTransformationArtifact:
        ti = kwargs["ti"]
        data_validation_artifact = ti.xcom_pull(task_ids = "data_validation", key= "data_validation_artifact")
        # data_validation_artifact = DataValidationArtifact(data_validation_artifact)

        data_transformation_config = training_config.get_data_transformation_config()
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config
                                                    )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        ti.xcom_push('data_transformation_artifact',data_transformation_artifact)

    def training_model(**kwargs) -> ModelTrainerArtiact:
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids = "data_transformation", key= "data_transformation_artifact")
        # data_transformation_artifact = DataTransformationArtifact(data_transformation_artifact)

        model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                        model_trainer_config= training_config.get_model_trainer_config()
                                        )
        model_trainer_artifact = model_trainer.initiate_model_training()
        ti.xcom_push("model_trainer_artifact",model_trainer_artifact)
        
    def evaluating_model(**kwargs) -> ModelEvaluationArtifact:
        
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids = "data_transformation", key= "data_transformation_artifact")
        # data_transformation_artifact = DataTransformationArtifact(data_transformation_artifact)
        model_trainer_artifact = ti.xcom_pull(task_ids = "model_trainer", key= "model_trainer_artifact")
        # model_trainer_artifact = ModelTrainerArtiact(model_trainer_artifact)
        

        model_eval_config = training_config.get_model_evaluation_config()
        model_eval = ModelEvaluation(data_transformation_artifact=data_transformation_artifact,
                                        model_trainer_artifact=model_trainer_artifact,
                                        model_evaluation_config=model_eval_config
                                        )
        model_evalutaion_artifact = model_eval.initiate_model_evalution()
        ti.xcom_push("model_evalutaion_artifact",model_evalutaion_artifact)
        
    def pushing_model(**kwargs):
        ti = kwargs["ti"]
        model_evalutaion_artifact = ti.xcom_pull(task_ids = "model_evaluation", key= "model_evalutaion_artifact")
        data_ingestion_artifact = ti.xcom_pull(task_ids = "data_ingestion", key= "data_ingestion_artifact")

        model_pusher_config = training_config.get_model_pusher_config()
        model_pusher = ModelPusher(model_evaluation_artifact=model_evalutaion_artifact,
                                    model_pusher_config=model_pusher_config
                                    )
        model_pusher_artifact =  model_pusher.initiate_model_pusher()
        ti.xcom_push("model_evalutaion_artifact",model_pusher_artifact)

        data_ingestion_config = training_config.get_dataingestion_config()
        update_recent_batch_in_meta_data(metadata_file_path=data_ingestion_config.meta_data_file_path,
                                            newly_downloaded_batches=data_ingestion_artifact.new_batches_timestamps)
        save_artifacts_to_s3_and_clear_local()
                


    ingesting_data = PythonOperator(
        task_id='data_ingestion',
        python_callable=ingesting_data,
    )


    validating_data = PythonOperator(
        task_id="data_validation",
        python_callable=validating_data

    )

    transforming_data = PythonOperator(
        task_id ="data_transformation",
        python_callable=transforming_data
    )

    training_model = PythonOperator(
        task_id="model_trainer", 
        python_callable=training_model

    )

    evaluating_model = PythonOperator(
        task_id="model_evaluation", python_callable=evaluating_model
    )   

    pushing_model  =PythonOperator(
            task_id="push_model",
            python_callable=pushing_model

    )

    ingesting_data >> validating_data >> transforming_data >> training_model >> evaluating_model >> pushing_model