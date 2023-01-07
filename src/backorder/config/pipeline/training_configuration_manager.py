from backorder.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from backorder.entity import ModelPusherConfig
from backorder.exception import BackorderException
from backorder.logger import logging
import sys,os
from backorder.constants.training_pipeline_config import *
from backorder.constants.aws_s3 import *
from datetime import datetime
from pathlib import Path


class TrainingConfigurationManager:

    def __init__(self, timestamp = None ):
        try:
            self.timestamp = timestamp if timestamp != None else TIMESTAMP
            self.artifact_dir = Path(os.path.join(ARTIFACT_DIR, self.timestamp))
        except Exception as e:
            logging.info(BackorderException(e,sys))
    
    def get_dataingestion_config(self):
        try:
            data_ingestion_artifact_dir = Path(os.path.join(self.artifact_dir,DATA_INGESTION_DIR_NAME))
            feature_store_dir = Path(os.path.join(data_ingestion_artifact_dir, DATA_INGESTION_FEATURE_STORE_DIR))
            feature_store_raw_data_dir = Path(os.path.join(feature_store_dir,DATA_INGESTION_FEATURE_STORE_RAW_DATA_DIR_NAME))
            feature_store_merged_filePath = Path(os.path.join(feature_store_dir,
                                                                DATA_INGESTION_FEATURE_STORE_MERGED_DATA_DIR_NAME,
                                                                DATA_INGESTION_FEATURE_STORE_FILENAME))
                                                                
            good_data_dir = Path(os.path.join(data_ingestion_artifact_dir,DATA_INGESTION_GOOD_DATA_DIR))
            bad_data_dir = Path(os.path.join(data_ingestion_artifact_dir,DATA_INGESTION_BAD_DATA_DIR))

            meta_data_file_path= Path(os.path.join(data_ingestion_artifact_dir, DATA_INGESTION_METADATA_DIR_NAME,
                                                    DATA_INGESTION_METADATA_FILE_NAME))

            data_ingestion_config = DataIngestionConfig(
                                                    aws_data_source_bucket_name= DATA_SOURCE_BUCKET_NAME,
                                                    aws_artifacts_bucket_name= ARTIFACTS_BUCKET_NAME,
                                                    feature_store_dir= feature_store_dir,
                                                    feature_store_raw_data_dir= feature_store_raw_data_dir,
                                                    feature_store_merged_filePath= feature_store_merged_filePath, 
                                                    meta_data_file_path=meta_data_file_path,
                                                    data_ingestion_dir_name=DATA_INGESTION_DIR_NAME,
                                                    source_data_file_name=DATA_INGESTION_SOURCE_FILE_NAME,
                                                    bad_data_dir= bad_data_dir,
                                                    good_data_dir= good_data_dir
                                                )
            return data_ingestion_config
        except Exception as e:
            logging.info(BackorderException(e, sys))

    def get_data_validatin_config(self):

        try:
            data_validation_artifact_dir = Path(os.path.join(self.artifact_dir, DATA_VALIDATION_ARTIFACT_DIR_NAME))
            valid_data_dir = Path(os.path.join(data_validation_artifact_dir,DATA_VALIDATION_VALID_DATA_DIR_NAME))
            invalid_data_dir = Path(os.path.join(data_validation_artifact_dir, DATA_VALIDATION_INVALID_DATA_DIR_NAME))
            valid_train_file_path= Path(os.path.join(valid_data_dir,DATA_VALIDATION_VALID_TRAIN_FILE_NAME))
            invalid_train_file_path = Path(os.path.join(invalid_data_dir,DATA_VALIDATION_INVALID_TRAIN_FILE_NAME))
            valid_test_file_path = Path(os.path.join(valid_data_dir, DATA_VALIDATION_VALID_TEST_FILE_NAME))
            invalid_test_file_path = Path(os.path.join(invalid_data_dir, DATA_VALIDATION_INVALID_TEST_FILE_NAME))
            valid_validation_file_path= Path(os.path.join(valid_data_dir, DATA_VALIDATION_VALID_VALIDATION_FILE_NAME))
            invalid_validation_file_path= Path(os.path.join(invalid_data_dir, DATA_VALIDATION_INVALID_VALIDATION_FILE_NAME))
            drift_report_file_path = Path(os.path.join(data_validation_artifact_dir,
                                                            DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
                                                            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME))

            data_validation_config = DataValidationConfig(
                                            data_validation_artifact_dir= data_validation_artifact_dir,
                                            drift_report_file_path= drift_report_file_path,
                                            invalid_data_dir=invalid_data_dir ,
                                            invalid_test_file_path=invalid_test_file_path,
                                            invalid_train_file_path=invalid_train_file_path,
                                            ks_2_samp_test_threshold= DATA_VALIDATION_KS_2SAMP_TEST_THRESHOLD,
                                            valid_data_dir=valid_data_dir,
                                            test_split_ratio= DATA_VALIDATION_TEST_SPLIT_RATIO,
                                            validation_split_ratio= DATA_VALIDATION_VALIDATION_SPLIT_RATIO,
                                            valid_test_file_path=valid_test_file_path,
                                            valid_train_file_path=valid_train_file_path,
                                            valid_validation_file_path= valid_validation_file_path,
                                            invalid_validation_file_path=invalid_validation_file_path
            )
            return data_validation_config

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise(BackorderException(e,sys))

    def get_data_transformation_config(self):
        try:

            data_transformation_artifact_dir = Path(os.path.join(self.artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR_NAME))
            preprocessor_obj_file_path = Path(os.path.join(data_transformation_artifact_dir,
                                                            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                            DATA_TRANSFORMATION_PREPROCESSOR_OBJ_FILE_NAME))
            transformed_data_dir = Path(os.path.join(data_transformation_artifact_dir, DATA_TRANSFORMATION_TRASNFORMED_DIR_NAME))
            transformed_test_file_path = Path(os.path.join(transformed_data_dir, DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME))
            transformed_train_file_path = Path(os.path.join(transformed_data_dir, DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME))
            transformed_valid_file_path = Path(os.path.join(transformed_data_dir,DATA_TRANSFORMATION_TRANSFORMED_VALID_FILE_NAME))

            data_transformation_config = DataTransformationConfig(data_transformation_artifact_dir= data_transformation_artifact_dir,
                                                                    preprocessor_obj_file_path=preprocessor_obj_file_path,
                                                                    transformed_test_file_path= transformed_test_file_path,
                                                                    transformed_train_file_path= transformed_train_file_path,
                                                                    transformed_valid_file_path=transformed_valid_file_path
                                                                    
                                                                    )
            return data_transformation_config
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise(BackorderException(e,sys))


    def get_model_trainer_config(self):
        try:
            model_trainer_artifact_dir = Path(os.path.join(self.artifact_dir,MODEL_TRAINER_ARTIFACT_DIR_NAME))
            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
                                                    MODEL_TRAINER_TRAINED_MODEL_DIR_NAME,
                                                    MODEL_TRAINER_TRAINED_MODEL_NAME)

            model_trainer_config = ModelTrainerConfig(
                                            trained_model_file_path=trained_model_file_path,
                                            models_config_file_path=MODEL_TRAINER_MODELS_CONFIG_FILE_PATH,
                                            base_accuracy= MODEL_TRAINER_BASE_ACCURACY
                                            )
            return model_trainer_config
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise(BackorderException(e,sys))
        
    def get_model_evaluation_config(self):
        try:
            model_evaluation_artifact_dir = Path(os.path.join(self.artifact_dir, MODEL_EVALUATION_ARTIFACT_DIR_NAME))
            best_model_artifact_file_path = Path(os.path.join(model_evaluation_artifact_dir,"model_registry", BEST_MODEL_OBJ_FILE_NAME))
            accepted_model_dir = Path(os.path.join(model_evaluation_artifact_dir,MODEL_EVALUATION_ACCEPTED_MODEL_DIR_NAME))
            accpeted_model_artifact_file_path = Path(os.path.join(accepted_model_dir, BEST_MODEL_OBJ_FILE_NAME))
            model_registry_bucket_latest_model_key = MODEL_REGISTRY_LATEST_MODEL_DIR_NAME+"/"+BEST_MODEL_OBJ_FILE_NAME

            model_evaluation_config = ModelEvaluationConfig(
                                                changed_accuracy_threshold= MODEL_EVALUATION_CHANGED_THRESHOLD_ACCURACY,
                                                model_registry_bucket_latest_model_key = model_registry_bucket_latest_model_key,
                                                model_registry_latest_model_dir= MODEL_REGISTRY_LATEST_MODEL_DIR_NAME,
                                                best_model_artifact_file_path =best_model_artifact_file_path,
                                                accepted_model_artifact_file_path= accpeted_model_artifact_file_path,
                                                model_registry_bucket_name = MODEL_REGISTRY_BUCKET_NAME,
                                                best_model_obj_file_name= BEST_MODEL_OBJ_FILE_NAME,

                                                )
            return model_evaluation_config

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise(BackorderException(e,sys))

    def get_model_pusher_config(self):
        try:
            
            model_pusher_config = ModelPusherConfig(
                                                model_registry_bucket_name= MODEL_REGISTRY_BUCKET_NAME,
                                                model_registry_latest_model_dir_key= MODEL_REGISTRY_LATEST_MODEL_FOLDER_NAME,
                                                model_registry_latest_model_obj_key= MODEL_REGISTRY_LATEST_MODEL_KEY,
                                                model_registry_previous_model_obj_key= MODEL_REGISTRY_PREVIOUS_MODEL_KEY,
                                                model_registry_previous_model_dir = MODEL_REGISTRY_PREVIOUS_MODEL_DIR_NAME
                                                )
            return model_pusher_config
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise(BackorderException(e,sys))