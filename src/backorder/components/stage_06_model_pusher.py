from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import ModelPusherConfig
from backorder.entity.artifact_entity import  ModelEvaluationArtifact,ModelPusherArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from backorder.cloud_storage.s3_operations import S3Operations
from backorder.utils import load_object, create_directories
import shutil
from backorder.data_access import MongodbOperations


class ModelPusher:
    def __init__(self, model_evaluation_artifact:ModelEvaluationArtifact,
                        model_pusher_config: ModelPusherConfig):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            self.s3_operations = S3Operations()
            self.mongo_operations = MongodbOperations()

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)


    def initiate_model_pusher(self):

        try:
            if self.model_evaluation_artifact.is_model_accepted:
                accepted_model_file_path = self.model_evaluation_artifact.accepted_model_path
                best_model_path = self.model_evaluation_artifact.best_model_path
                logging.info(f"loading the accepted model from {accepted_model_file_path}")
                accepted_model_obj = load_object(file_path= accepted_model_file_path)
                accepted_model_dir = os.path.dirname(accepted_model_file_path)
                latest_model_dir = self.model_pusher_config.model_registry_latest_model_dir_key
                aws_latest_model_dir_url = f"s3://{self.model_pusher_config.model_registry_bucket_name}/{latest_model_dir}"
                previous_model_dir = self.model_pusher_config.model_registry_previous_model_dir
                aws_previous_model_dir_url = f"s3://{self.model_pusher_config.model_registry_bucket_name}/{previous_model_dir}"

                if best_model_path is None:
                    self.s3_operations.sync_folder_to_s3(folder=accepted_model_dir, aws_bucket_url=aws_latest_model_dir_url)
                
                else:
                    logging.info(f"moving the latest model in model registry to the previous folder")
                    best_model_dir = os.path.dirname(best_model_path)
                    self.s3_operations.sync_folder_to_s3(folder= best_model_dir, aws_bucket_url= aws_previous_model_dir_url)
                    
                    logging.info(f"now syncing the accepted model to modelregistry's latest folder")
                    self.s3_operations.sync_folder_to_s3(folder=accepted_model_dir, aws_bucket_url=aws_latest_model_dir_url)

                model_pusher_artifact = ModelPusherArtifact(training_phase= "Model_Pusher",
                                                            is_model_pushed=True,
                                                            bucket_name= self.model_pusher_config.model_registry_bucket_name,
                                                            latest_model_obj_key= self.model_pusher_config.model_registry_latest_model_obj_key)
                logging.info(f"model_pusher_artifact: {model_pusher_artifact}")
                
                model_pusher_artifact_dict = model_pusher_artifact.__dict__
                self.mongo_operations.save_artifact(artifact= model_pusher_artifact_dict)
                logging.info(f"saved the model_pusher artifact to mongodb") 
                return model_pusher_artifact
            else:
                logging.info(f"since trained model is not accepted in evaluation phase, now not initiating the model phuser to push the trained model into model registry")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)