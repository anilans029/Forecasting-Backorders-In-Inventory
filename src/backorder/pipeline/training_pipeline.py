from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
from backorder.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from backorder.entity.config_entity import ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from backorder.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from backorder.entity.artifact_entity import ModelTrainerArtiact, ModelEvaluationArtifact, ModelPusherArtifact
from backorder.components import DataIngestion, DataValidation, DataTransformation, ModelTrainer, ModelEvaluation,ModelPusher
from backorder.exception import BackorderException
from backorder.logger import logging
import os,sys
from backorder.utils import save_artifacts_to_s3_and_clear_local, update_recent_batch_in_meta_data


class TrainingPipeline:

    def __init__(self, training_config: TrainingConfigurationManager = TrainingConfigurationManager()):
        self.training_config= training_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.training_config.get_dataingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact, data_ingestion_config

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.training_config.get_data_validatin_config()
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = self.training_config.get_data_transformation_config()
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config

                                                     )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtiact:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.training_config.get_model_trainer_config()
                                         )
            model_trainer_artifact = model_trainer.initiate_model_training()
            return model_trainer_artifact
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_model_evaluation(self, data_transformation_artifact, model_trainer_artifact) -> ModelEvaluationArtifact:
        try:
            model_eval_config = self.training_config.get_model_evaluation_config()
            model_eval = ModelEvaluation(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_artifact=model_trainer_artifact,
                                         model_evaluation_config=model_eval_config
                                         )
            return model_eval.initiate_model_evalution()
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            model_pusher_config = self.training_config.get_model_pusher_config()
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=model_pusher_config
                                       )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start(self):
        try:
            data_ingestion_artifact,data_ingestion_config = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_transformation_artifact=data_transformation_artifact,
                                                              model_trainer_artifact=model_trainer_artifact
                                                              )
            if model_eval_artifact.is_model_accepted:
                self.start_model_pusher(model_evaluation_artifact=model_eval_artifact)
            update_recent_batch_in_meta_data(metadata_file_path=data_ingestion_config.meta_data_file_path,
                                            newly_downloaded_batches=data_ingestion_artifact.new_batches_timestamps)
            save_artifacts_to_s3_and_clear_local()
            

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)
