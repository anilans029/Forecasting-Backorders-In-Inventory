from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import ModelEvaluationConfig
from backorder.entity.artifact_entity import ModelTrainerArtiact, ModelEvaluationArtifact,DataTransformationArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config import *
from backorder.cloud_storage.s3_operations import S3Operations
from backorder.utils import load_object, create_directories, load_numpy_array_data
import shutil
from backorder.ml.metric import EvaluateClassificationModel


class ModelEvaluation:
    def __init__(self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtiact,
        data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_evaluation_config= model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.s3_operations = S3Operations()
        
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def sync_the_latest_model_from_model_registry(self,folder,bucket_name,latest_model_dir):
        try:
            
            best_model_artifact_dir = os.path.dirname(folder)
            create_directories([best_model_artifact_dir])
            aws_buckek_url = f"s3://{bucket_name}/{latest_model_dir}"
            self.s3_operations.sync_s3_to_folder(folder= best_model_artifact_dir, aws_bucket_url= aws_buckek_url) 
            logging.info(f"succefully synced the best model from model registry to this machine")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
    def initiate_model_evalution(self):
        try:
            if self.model_trainer_artifact.is_model_found:
                logging.info(f"{'*'*10} initiating the Model Evaluation Phase {'*'*10}\n")
                trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
                best_model_artifact_file_path = self.model_evaluation_config.best_model_artifact_file_path
                model_registry_bucket_name = self.model_evaluation_config.model_registry_bucket_name
                model_registry_latest_model_dir = self.model_evaluation_config.model_registry_latest_model_dir
                model_registry_latest_model_key = self.model_evaluation_config.model_registry_bucket_latest_model_key
                threshold_accuracy = self.model_evaluation_config.changed_accuracy_threshold
                accepted_model_file_path = self.model_evaluation_config.accepted_model_artifact_file_path

                logging.info(f"loading the train and test datasets ")
                train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
                test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
                train_x = train_arr[:,:-1]
                train_y =train_arr[:,-1]
                test_x= test_arr[:,:-1]
                test_y=test_arr[:, -1]
                
                logging.info(f"loading the trained model object")
                trained_best_model_obj = load_object(trained_model_file_path)

                logging.info(f"checking if there is latest model available in model registry")
                is_latest_model_available = self.s3_operations.is_s3_key_path_available(bucket_name=model_registry_bucket_name,
                                                                s3_key= model_registry_latest_model_key)
                if is_latest_model_available:
                    logging.info(f"best model available in the model registry. lets get that model obj for evaluation")
                    self.sync_the_latest_model_from_model_registry(bucket_name= model_registry_bucket_name,
                                                                    folder = best_model_artifact_file_path,
                                                                    latest_model_dir= model_registry_latest_model_dir)

                    logging.info(f"loading the best model from model_registry")
                    best_model_obj = load_object(file_path=best_model_artifact_file_path)

                    evaluatate_classification_model = EvaluateClassificationModel(train_x= train_x,
                                                                                train_y= train_y,
                                                                                test_x= test_x,
                                                                                test_y= test_y
                                                                                )

                    trained_model_metrics = evaluatate_classification_model.evaluate_and_get_classification_metric(
                                                                                        model=trained_best_model_obj
                                                                                        )
                    trained_model_accuracy = trained_model_metrics.model_accuracy
                    best_model_metrics = evaluatate_classification_model.evaluate_and_get_classification_metric(
                                                                                            model = best_model_obj
                                                                                        )
                    best_model_accuracy = best_model_metrics.model_accuracy
                    diff_accuracy = abs(trained_model_accuracy - best_model_accuracy)
                    logging.info(f"the difference betweent the trained and best model accuracies is {diff_accuracy}")
                    
                    
                    if (trained_model_accuracy> best_model_accuracy) and (diff_accuracy > threshold_accuracy):
                        logging.info(f"trained model accuracy is more than the best model accuracy. so accepting the trained model")
                        create_directories([os.path.dirname(accepted_model_file_path)])
                        shutil.copy(trained_model_file_path,accepted_model_file_path)

                        model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted= True,
                                                                            changed_accuracy= diff_accuracy,
                                                                            best_model_metric_artifact= trained_model_metrics,
                                                                            accepted_model_path = accepted_model_file_path,
                                                                            best_model_path= best_model_artifact_file_path,
                                                                            )
                        logging.info(f"model_evaluation_artifact : {model_evaluation_artifact}")
                        return model_evaluation_artifact
                    else:
                        logging.info(f"trained model accuracy is less than the best model accuracy. so rejecting the trained model")
                        model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted= False,
                                                                            changed_accuracy= None,
                                                                            best_model_metric_artifact= best_model_metrics,
                                                                            accepted_model_path = None,
                                                                            best_model_path= best_model_artifact_file_path,
                                                                            )
                        logging.info(f"model_evaluation_artifact : {model_evaluation_artifact}")
                        return model_evaluation_artifact
                else:

                    logging.info(f"Latest model is not available in the model registry. So, accepting the trained model")
                    ## initialized the EvaluateClassificationModel with train and test datasets
                    evaluatate_classification_model = EvaluateClassificationModel(train_x= train_x,
                                                                                train_y= train_y,
                                                                                test_x= test_x,
                                                                                test_y= test_y
                                                                                )
                    trained_model_metrics = evaluatate_classification_model.evaluate_and_get_classification_metric(
                                                                                        model=trained_best_model_obj
                                                                                        )
                    trained_model_accuracy = trained_model_metrics.model_accuracy
                    logging.info(f"saving the accepted model obj at {accepted_model_file_path} ")
                    create_directories([os.path.dirname(accepted_model_file_path)])
                    shutil.copy(trained_model_file_path,accepted_model_file_path)
                    model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted= True,
                                                                        changed_accuracy= None,
                                                                        best_model_metric_artifact= trained_model_metrics,
                                                                        accepted_model_path = accepted_model_file_path,
                                                                        best_model_path= None,
                                                                        )
                    logging.info(f"model_evaluation_artifact : {model_evaluation_artifact}")
                    logging.info(f"{'*'*10} Model Evaluation Phase completed {'*'*10}\n")
                    return model_evaluation_artifact
            else:
                raise(f"since best Model is not found in Training phase, not initiating the model Evaluation phase")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)