from backorder.entity import DataIngestionConfig
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
                                                    test_split_ratio= DATA_INGESTION_TEST_SPLIT_RATIO,
                                                    failed_data_ingestion_dir = failed_ingestion_dir,
                                                    meta_data_file_path=meta_data_file_path,
                                                    data_ingestion_dir_name=DATA_INGESTION_DIR_NAME,
                                                    source_data_file_name=DATA_INGESTION_SOURCE_FILE_NAME
                                                )
          
            return data_ingestion_config
            
        except Exception as e:
            logging.info(BackorderException(e, sys))