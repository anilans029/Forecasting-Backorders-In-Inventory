from backorder.logger import logging
from backorder.exception import BackorderException
import os, sys
from backorder.config import TrainingConfigurationManager
from backorder.components import DataIngestion
# from backorder.cloud_storage.s3_operations import S3Operations

# s3_oper = S3Operations()
# buckets = s3_oper.list_all_buckets_in_s3()
# print(buckets)

# objects = s3_oper.list_all_objects_in_s3Bucket("truck-aps-sensor-pipeline")
# # lt_obj = [obj.split("/")[0] for obj in objects if len(obj.split("/"))==2]

# print(objects)


training_config = TrainingConfigurationManager()
data_ingestion_config = training_config.get_dataingestion_config()
data_ingestion = DataIngestion(config=data_ingestion_config)
data_ingestion.initiate_data_ingestion()