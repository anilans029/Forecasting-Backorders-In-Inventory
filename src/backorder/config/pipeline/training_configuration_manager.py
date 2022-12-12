from backorder.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from backorder.exception import BackorderException
from backorder.logger import logging
import sys,os
from backorder.constants.training_pipeline_config import *
from backorder.constants.aws_s3 import DATA_SOURCE_BUCKET_NAME, ARTIFACTS_BUCKET_NAME
from datetime import datetime
from pathlib import Path


class TrainingConfigurationManager:

    def __init__(self, timestamp= datetime.now()):
        try:
            self.timestamp = timestamp.strftime("%d-%m-%Y__%H-%M-%S")
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
            ingested_data_dir = Path(os.path.join(data_ingestion_artifact_dir, DATA_INGESTION_INGESTED_DIR))
            failed_ingestion_dir = Path(os.path.join(data_ingestion_artifact_dir, DATA_INGESTION_FAILED_INGESTION_DIR))
            train_file_path= Path(os.path.join(ingested_data_dir, DATA_INGESTION_TRAIN_FILE_PATH))
            test_file_path = Path(os.path.join(ingested_data_dir, DATA_INGESTION_TEST_FILE_PATH))
            validation_file_path = Path(os.path.join(ingested_data_dir, DATA_INGESTION_VALIDATION_FILE_NAME))
            meta_data_file_path= Path(os.path.join(data_ingestion_artifact_dir, DATA_INGESTION_METADATA_DIR_NAME,
                                                    DATA_INGESTION_METADATA_FILE_NAME))

            data_ingestion_config = DataIngestionConfig(
                                                    aws_data_source_bucket_name= DATA_SOURCE_BUCKET_NAME,
                                                    aws_artifacts_bucket_name= ARTIFACTS_BUCKET_NAME,
                                                    ingested_data_dir= ingested_data_dir,
                                                    feature_store_dir= feature_store_dir,
                                                    feature_store_raw_data_dir= feature_store_raw_data_dir,
                                                    feature_store_merged_filePath= feature_store_merged_filePath,
                                                    train_file_path= train_file_path,
                                                    test_file_path= test_file_path,
                                                    validation_file_path=validation_file_path,
                                                    test_split_ratio= DATA_INGESTION_TEST_SPLIT_RATIO,
                                                    validation_split_ratio= DATA_INGESTION_VALIDATION_SPLIT_RATIO, 
                                                    failed_data_ingestion_dir = failed_ingestion_dir,
                                                    meta_data_file_path=meta_data_file_path,
                                                    data_ingestion_dir_name=DATA_INGESTION_DIR_NAME,
                                                    source_data_file_name=DATA_INGESTION_SOURCE_FILE_NAME
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