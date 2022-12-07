from backorder.logger import logging
from backorder.exception import BackorderException
from backorder.constants.aws_s3 import *
from backorder.entity.artifact_entity import DataIngestionArtifact
from backorder.entity.config_entity import DataIngestionConfig
from backorder.constants.training_pipeline_config import *
from backorder.config import TrainingConfigurationManager
from backorder.cloud_storage.s3_operations import S3Operations
from backorder.utils import create_directories
import os, sys

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.data_ingestion_config = config
            self.s3_operations = S3Operations()

        except Exception as e:
            logging.info(BackorderException(e, sys))

    def get_updated_timestamp_and_sync_from_s3(self):
        try:
            meta_info_file_name = self.data_ingestion_config.meta_data_file_name
            folder_name= self.data_ingestion_config.data_ingestion_dir_name
            feature_store_dir_name = os.path.basename(self.data_ingestion_config.feature_store_dir)
            merged_dir = os.path.dirname(self.data_ingestion_config.feature_store_merged_filePath)
            meta_data_dir = os.path.dirname(self.data_ingestion_config.feature_store_merged_filePath)
            merged_dir_name = os.path.basename(os.path.dirname(self.data_ingestion_config.feature_store_merged_filePath))
            merged_data_filename = os.path.basename(self.data_ingestion_config.feature_store_merged_filePath)
            artifacts_bucket_name = self.data_ingestion_config.aws_artifacts_bucket_name

            logging.info(f"checking if objects are availbe in {artifacts_bucket_name} bucket")
            available_objects = self.s3_operations.list_all_objects_in_s3Bucket(artifacts_bucket_name)
            if available_objects!= None:
                logging.info(f"objects are availbe in {artifacts_bucket_name} bucket. so, now getting all the timestamps")
                print(available_objects)
                timestamps= [obj.split("/")[0] for obj in available_objects if len(obj.split('/'))==2]
                print(timestamps)
                timestamps.sort(reverse=True)
                latest_timestamp = timestamps[0]
                logging.info(f"retreived the latest timestamp {latest_timestamp} from the objects folders")
                
                aws_bucket_url_meta_info = f"s3://{artifacts_bucket_name}/{latest_timestamp}/{folder_name}/{meta_info_file_name}"
                aws_bucket_url_feature_store = f"""s3://{artifacts_bucket_name}/
                                                    {latest_timestamp}/{folder_name}/
                                                    {feature_store_dir_name}/
                                                    {merged_dir_name}/{merged_data_filename}"""
                create_directories([merged_dir, meta_data_dir])
                logging.info(f"syncing the s3 to folder: {meta_data_dir}")
                self.s3_operations.sync_s3_to_folder(meta_data_dir, aws_bucket_url_meta_info)
                logging.info(f"syncing the s3 to folder: {merged_dir}")
                self.s3_operations.sync_s3_to_folder(merged_dir, aws_bucket_url_feature_store)



            else:
                logging.info("The s3 bucket: {artifacts_bucket_name} doesn't contain any objects")

        except Exception as e:
            logging.info(BackorderException(e, sys))

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("initiating the data ingestion")
            self.get_updated_timestamp_and_sync_from_s3()

        except Exception as e:
            logging.info(BackorderException(e, sys))