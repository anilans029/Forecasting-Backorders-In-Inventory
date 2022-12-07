from backorder.logger import logging
from backorder.exception import BackorderException
from backorder.constants.aws_s3 import *
from backorder.entity.artifact_entity import DataIngestionArtifact
from backorder.entity.config_entity import DataIngestionConfig
from backorder.constants.training_pipeline_config import *
from backorder.config import TrainingConfigurationManager
from backorder.cloud_storage.s3_operations import S3Operations
from backorder.utils import create_directories,read_yaml, write_yaml
import os, sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.data_ingestion_config = config
            self.s3_operations = S3Operations()
            self.schema_file = read_yaml(filepath= os.path.join("src","backorder","config","schema.yaml"))

        except Exception as e:
            logging.info(BackorderException(e, sys))

    def get_updated_timestamp_and_sync_from_s3(self):
        try:
            meta_info_dir = os.path.dirname(self.data_ingestion_config.meta_data_file_path)
            meta_info_dir_name = os.path.basename(meta_info_dir)
            folder_name= self.data_ingestion_config.data_ingestion_dir_name
            feature_store_dir_name = os.path.basename(self.data_ingestion_config.feature_store_dir)
            merged_dir = os.path.dirname(self.data_ingestion_config.feature_store_merged_filePath)
            merged_dir_name = os.path.basename(os.path.dirname(self.data_ingestion_config.feature_store_merged_filePath))
            artifacts_bucket_name = self.data_ingestion_config.aws_artifacts_bucket_name


            logging.info(f"checking if objects are availbe in {artifacts_bucket_name} bucket")
            available_objects = self.s3_operations.list_all_objects_in_s3Bucket(artifacts_bucket_name)
            if available_objects!= None:
                logging.info(f"objects are availbe in {artifacts_bucket_name} bucket. so, now getting all the timestamps")
                timestamps= [obj.split("/")[0] for obj in available_objects if len(obj.split('/'))==2]
                timestamps.sort(reverse=True)
                latest_timestamp = timestamps[0]
                logging.info(f"retreived the latest timestamp {latest_timestamp} from the objects folders")
                
                aws_bucket_url_meta_info = f"s3://{artifacts_bucket_name}/{latest_timestamp}/{folder_name}/{meta_info_dir_name}"
                aws_bucket_url_feature_store = f"s3://{artifacts_bucket_name}/{latest_timestamp}/{folder_name}/{feature_store_dir_name}/{merged_dir_name}"
                create_directories([merged_dir, meta_info_dir])
                logging.info(f"syncing from [{aws_bucket_url_meta_info}] to folder: {meta_info_dir}")
                self.s3_operations.sync_s3_to_folder(meta_info_dir, aws_bucket_url_meta_info)
                logging.info(f"syncing from [{aws_bucket_url_feature_store}] to folder: {merged_dir}")
                self.s3_operations.sync_s3_to_folder(merged_dir, aws_bucket_url_feature_store)

                meta_data_dict = read_yaml(self.data_ingestion_config.meta_data_file_path)
                return datetime.strptime(meta_data_dict["recently_used_batch_date"],"%d-%m-%Y")

            else:
                logging.info("The s3 bucket: {artifacts_bucket_name} doesn't contain any objects")

        except Exception as e:
            logging.info(BackorderException(e, sys))

    def get_latest_batch_date_from_source(self):
        try:
            logging.info(f"checking if objects available in the data_source_bucket : {self.data_ingestion_config.aws_data_source_bucket_name}")
            available_objects = self.s3_operations.list_all_objects_in_s3Bucket(bucket_name=self.data_ingestion_config.aws_data_source_bucket_name)
            if len(available_objects)>0:
                batch_dates = [obj.split("/")[0] for obj in available_objects if len(obj.split("/"))==2]
                batch_dates.sort(reverse= True)
                recent_batch_date_from_data_source = datetime.strptime(batch_dates[0], "%d-%m-%Y")
                batch_dates=  list(map(lambda x : datetime.strptime(x, "%d-%m-%Y"), batch_dates))
                return recent_batch_date_from_data_source,batch_dates
            else:
                logging.info(f"there are no objects present at the data-source: {self.data_ingestion_config.aws_data_source_bucket_name}")

        except Exception as e:
            logging.info(BackorderException(e,sys))


    def download_data_from_source_bucket(self, date:datetime):
        try:
            date = date.strftime("%d-%m-%Y")
            download_file_path = Path(os.path.join(self.data_ingestion_config.feature_store_raw_data_dir,
                                                    date,
                                                    self.data_ingestion_config.source_data_file_name))
            create_directories([os.path.dirname(download_file_path)])
            object_path = f"{date}/{self.data_ingestion_config.source_data_file_name}"

            self.s3_operations.download_file_from_s3(bucket_name= self.data_ingestion_config.aws_data_source_bucket_name,
                                                    object_path=object_path,
                                                    download_file_path=str(download_file_path))
            logging.info(f"successfully downloaded the {date}/{self.data_ingestion_config.source_data_file_name} and stored at :{download_file_path}")
            return True
        except Exception as e:
            logging.info(BackorderException(e,sys))

    def download_all_new_data_from_source_bucket(self,new_batch_dates: list):
        try:
            downloaded_batches = []
            download_failed_batches = []
            for dt in new_batch_dates:
                    retries = 2
                    if self.download_data_from_source_bucket(date = dt):
                        downloaded_batches.append(dt)
                    else:
                        while(retries > 0):
                            logging.info(f"retrying to download the failed download file. retries_left={retries}")
                            if self.download_data_from_source_bucket(date=dt):
                                downloaded_batches.append(dt)
                                break
                            else:
                                retries -= 1
                        else:
                            logging.info(f"failed to downlaod the new_batch data from date: {dt}")
                            download_failed_batches.append(dt)
            return downloaded_batches
                

        except Exception as e:
            logging.info(BackorderException(e,sys))

    def convert_json_files_to_csv(self):
        try:
            raw_data_dir = self.data_ingestion_config.feature_store_raw_data_dir
            merging_df = pd.DataFrame()
            for i in os.listdir(raw_data_dir):
                new_file = Path(os.path.join(raw_data_dir,i,self.data_ingestion_config.source_data_file_name))
                if os.path.exists(new_file):
                    df_new = pd.read_json(new_file)
                    merging_df= pd.concat([merging_df,df_new])
            old_merged_filepath= self.data_ingestion_config.feature_store_merged_filePath
            if os.path.exists(old_merged_filepath):
                old_data_df = pd.read_csv(old_merged_filepath)
                merging_df = pd.concat([merging_df,old_data_df])
                logging.info(f"saving the merged df as csv file at: {[ self.data_ingestion_config.feature_store_merged_filePath ]}")
                merging_df.to_csv(self.data_ingestion_config.feature_store_merged_filePath)
                return merging_df
            else:
                merging_df = pd.concat([merging_df,old_data_df])
                logging.info(f"saving the merged df as csv file at: {[ self.data_ingestion_config.feature_store_merged_filePath ]}")
                merging_df.to_csv(self.data_ingestion_config.feature_store_merged_filePath)
                return merging_df
        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def split_data_to_train_test_sets(self,dataframe):
        try:
            train_set, test_set = train_test_split(dataframe, test_size= self.data_ingestion_config.test_split_ratio,
                                                    stratify= dataframe[self.schema_file["Target_column_Name"]])

            create_directories([self.data_ingestion_config.ingested_data_dir])
            logging.info(f"saving the train and test files at: {self.data_ingestion_config.ingested_data_dir}")
            train_set.to_csv(self.data_ingestion_config.train_file_path)
            test_set.to_csv(self.data_ingestion_config.test_file_path)

        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def write_meta_data(self,newly_downloaded_batches: list):
        try:
            newly_downloaded_batches.sort(reverse=True)
            recently_used_batch_date = newly_downloaded_batches[0].strftime("%d-%m-%Y")
            content = {"recently_used_batch_date": recently_used_batch_date}
            write_yaml(content= content, file_path= self.data_ingestion_config.meta_data_file_path)
        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("initiating the data ingestion")
            recent_used_batch_date = self.get_updated_timestamp_and_sync_from_s3()
            recent_batch_date_from_source, batch_dates = self.get_latest_batch_date_from_source()
            diff_date = recent_batch_date_from_source-recent_used_batch_date
            if diff_date.days>=1:
                logging.info("new data availabe in the data_source....lets download it")
                logging.info(f"getting the batch dates that are greater than recent_used_batch_date: {recent_used_batch_date}")
                new_batch_dates = [batch_date for batch_date in batch_dates if recent_used_batch_date < batch_date]
                downloaded_batches = self.download_all_new_data_from_source_bucket(new_batch_dates)
                
                logging.info(f"started converting the new raw json data into csv data by merging all new files")
                merged_df = self.convert_json_files_to_csv()
                
                logging.info(f"loading the merged csv file and splitting the data into train and test datasets")
                self.split_data_to_train_test_sets(dataframe= merged_df)

                logging.info(f"write the meta data into yaml file")
                self.write_meta_data(downloaded_batches)
                data_ingestion_artifact = DataIngestionArtifact(
                                                    feature_file_path= self.data_ingestion_config.feature_store_merged_filePath,
                                                    test_file_path= self.data_ingestion_config.test_file_path,
                                                    train_file_path= self.data_ingestion_config.train_file_path
                                            )
                return data_ingestion_artifact

            else:
                logging.info("no new data available in the data-source. so skipping the downloading fucntion")
                old_merged_filepath= self.data_ingestion_config.feature_store_merged_filePath
                if os.path.exists(old_merged_filepath):
                    old_data_df = pd.read_csv(old_merged_filepath)
                logging.info(f"loading the merged csv file and splitting the data into train and test datasets")
                self.split_data_to_train_test_sets(dataframe= old_data_df)
                data_ingestion_artifact = DataIngestionArtifact(
                                                    feature_file_path= self.data_ingestion_config.feature_store_merged_filePath,
                                                    test_file_path= self.data_ingestion_config.test_file_path,
                                                    train_file_path= self.data_ingestion_config.train_file_path
                                            )
                return data_ingestion_artifact
        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)