from backorder.exception import BackorderException
from backorder.logger import logging
import os,sys
from backorder.cloud_storage.s3_operations import S3Operations
from backorder.constants.aws_s3 import *
from backorder.constants.training_pipeline_config.schema_file_constants import *
from backorder.utils import read_model_byte_code,read_byte_coded_yaml_file,read_yaml
import pandas as pd
from backorder.ml.model.esitmator import TargetValueMapping
import json
from datetime import datetime

class PredictionPipeline:

    def __init__(self) -> None:
        try:
            self.s3_operations = S3Operations()
            self.s3_client = self.s3_operations.s3_client
            self.model_registry_bucket_name = MODEL_REGISTRY_BUCKET_NAME
            self.latest_model_s3_key = MODEL_REGISTRY_LATEST_MODEL_KEY
            self.prediction_bucket_name = PREDICTION_BUCKET_NAME
            self.prediction_batch_files_dir = PREDICTION_BATCHES_DIR_NAME
            self.predictions_outcome_dir = PREDICTION_OUTCOME_DIR_NAME
            self.meta_data_file_key = PREDICTION_METADATA_YAML_key
            self.target_value_mapping = TargetValueMapping()
            self.schema_of_data = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def is_latest_model_available(self):
        try:
            logging.info(f"checking if model_registry_bucket available or not")
            is_bucket_avaliable=  self.s3_operations.is_bucket_available_in_s3(bucket_name=self.model_registry_bucket_name)
            if is_bucket_avaliable:
                logging.info(f"the model-registry-bucket available in s3")
                logging.info(f"checking if the latest model_object available in bucket")
                is_model_key_path_available =self.s3_operations.is_s3_key_path_available(bucket_name=self.model_registry_bucket_name,
                                                            s3_key=self.latest_model_s3_key)
                if is_model_key_path_available:
                    logging.info(f"latest_model available in the model-registry")
                    return True
                else:
                    logging.info(f"latest-model is not available in the model-registry")
                    return False
            else:
                return False
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)


    def get_model_object(self):
        try:
            if self.is_latest_model_available():
                logging.info(f"getting the latest_model_obj from {self.model_registry_bucket_name}")
                response = self.s3_client.get_object(Bucket=self.model_registry_bucket_name,
                                        Key= self.latest_model_s3_key)
                model_byte_code = response["Body"].read()
                model_obj =  read_model_byte_code(model_byte_code)
                return model_obj

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def start_single_instance_prediction(self, dataframe:pd.DataFrame):
        try:
            #1. get the model from the s3 
            #2. now get the batch or single instance data in the form of dataframe from the s3 or front end
            #3. remove the unwanted columns
            #4. now transform the data using the transformer object by which we trained
            #5. now use the transformed data for doing the predictions
            #6. use the same model object to do predictions and replace the predicted outcome with the class "yes" or "no"
            #7. for single instance return the predictions
            #8. for the batch-data save the outcomes as csv and upload to s3 bucket
            model_object = self.get_model_object()
            logging.info(f"got the latest model object")

            prediction_outcome = model_object.transform_predict(dataframe)
            logging.info(f"prediction outcome :{int(prediction_outcome)}")
            predicion_class_name_dict =  self.target_value_mapping.reverse_mapping()
            final_pred_class = predicion_class_name_dict[int(prediction_outcome)]
            dataframe["went_on_backorder"] = final_pred_class
            logging.info(dataframe)
            # final_pred_df = (pd.DataFrame(prediction_outcome)).replace(predicion_classes_dict)
            if int(prediction_outcome)==0:
                logging.info(f"{final_pred_class}, it will not be backordered")
                return f"{final_pred_class}, it will not be backordered"
            else:
                logging.info(f"{final_pred_class}, it will be backordered")
                return f"{final_pred_class}, it will be backordered"

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def get_recently_predicted_batch_timestamp(self):
        try:
            logging.info(f"checking if meta_data yaml file available in the prediction s3 bucket")
            is_metadata_file_available = self.s3_operations.is_s3_key_path_available(
                                                        bucket_name= self.prediction_bucket_name,
                                                        s3_key= self.meta_data_file_key
                                                        )

            if is_metadata_file_available:
                logging.info(f" meta_data.yaml file is available in the s3 bucket. So, getting the file content")
                response = self.s3_client.get_object(Bucket= self.prediction_bucket_name,
                                          Key= self.meta_data_file_key)

                yaml_byte_code = response["Body"].read()
                yaml_dict = read_byte_coded_yaml_file(yaml_byte_code)
                return yaml_dict[RECENTLY_PREDICTED_BATCH_TIMESTAMP_key]
            else:
                logging.info(f"meta_info_not available in the prediction s3 bucket")
                return None
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys) 

    def get_all_timestamps_from_prediction_batch_files(self):
        try:
            objects_list =  self.s3_operations.list_all_objects_in_s3Bucket(
                            bucket_name='backorders-prediction-bucket',
                            prefix= 'prediction_batches')
            timestamps_list = [obj.split("/")[1] for obj in objects_list if (len(obj.split("/"))==3) and (obj.split("/")[2]!='')]
            return timestamps_list

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys) 
    
    def validate_data(self, dataframe: pd.DataFrame):
        try:
            if len(list(dataframe.columns)) == (self.schema_of_data[SCHEMA_FILE_TOTAL_NO_COLUMNS]):
                logging.info(f"total no.of columns are same as schema file")
                all_columns  =list(dataframe.columns)
                schema_all_cols = self.schema_of_data[SCHEMA_FILE_ALL_COLUMNS]
                schema_all_cols.remove("went_on_backorder")
                for col in schema_all_cols:
                    if col not in all_columns:
                        logging.info(f"few columns are not present in the batch file as per schema")
                        return False
                unwanted_cols = self.schema_of_data[SCHEMA_FILE_UNWANTED_COLUMNS]
                logging.info(f"unwanted_cols: {unwanted_cols}")
                dataframe.drop(columns=unwanted_cols, inplace=True)
                return True     
            else:
                logging.info('total no.of columns are not same as schema file')
                return False

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys) 

    def start_batch_prediction(self):
        try:
            logging.info(f"batch prediction initiatied....")
            logging.info(f"checking if the predictions batch files folder availabel in the prediction bucket")
            is_prediction_batch_files_dir_avaialable = self.s3_operations.is_s3_key_path_available(
                                                                        bucket_name= self.prediction_bucket_name,
                                                                        s3_key= self.prediction_batch_files_dir
                                                                        )
            if is_prediction_batch_files_dir_avaialable:
                logging.info(f"batch files folder available")
                recent_predicted_timestamp = datetime.strptime(str(self.get_recently_predicted_batch_timestamp()),"%d%m%y%H%M%S")
                
                logging.info("getting all the timestamps list in the prediction_batches folder")
                timestamps_list= self.get_all_timestamps_from_prediction_batch_files()
                
                if recent_predicted_timestamp!= None:
                    logging.info(f"checking if recent_predicted_timestamp is greater or lesser than the recent_timestamp in the timestamps_list ")
                    datetimes_list = [datetime.strptime(timestamp,"%d%m%y%H%M%S") for timestamp in timestamps_list]
                    datetimes_list.sort(reverse=True)
                    diff =   datetimes_list[0] - recent_predicted_timestamp
                else:
                    datetimes_list = [datetime.strptime(timestamp,"%d%m%y%H%M%S") for timestamp in timestamps_list]
                    datetimes_list.sort(reverse=True)
                if diff.days> 0 or recent_predicted_timestamp is None:
                    logging.info(f"new_prediction batches are availabel. so, getting all the timestamps after the recent_predicted_timestamp")
                    if recent_predicted_timestamp!= None:
                        new_timestamps = [datetime.strftime(date,"%d%m%y%H%M%S") for date in datetimes_list if recent_predicted_timestamp< date]
                    else:
                        new_timestamps = [datetime.strftime(date,"%d%m%y%H%M%S") for date in datetimes_list]
                    logging.info(f"new timestamps of prediction_batches: {new_timestamps}")

                    invalid_batches= []
                    predicted_batch_timestamps = []
                    for timestamp in new_timestamps:
                        new_batch_file_key = f"{self.prediction_batch_files_dir}/{timestamp}/{BACKORDERS_DATA_FILE_NAME}"
                        bucket_name = self.prediction_bucket_name 
                        logging.info(f"reading the csv file from s3://{bucket_name}/{new_batch_file_key}")
                        df = self.s3_operations.read_csv_file_from_s3(bucket_name= bucket_name,
                                                                        key= new_batch_file_key)
                        logging.info(f"got the batch data and converted to dataframe")
                        # logging.info(f"dropping not useful columns : [sku]")
                        # df.drop(columns=["sku"],inplace=True)
                        if self.validate_data(dataframe= df):
                            model_obj = self.get_model_object()
                            pred_arr = model_obj.transform_predict(df)
                            pred_df = pd.DataFrame(pred_arr,columns=["went_on_backorder"])
                            pred_df["went_on_backorder"] =pred_df["went_on_backorder"].replace(self.target_value_mapping.reverse_mapping())
                            df["went_on_backorder"] = pred_df["went_on_backorder"]
                            predicted_batch_file_key = f"{PREDICTION_OUTCOME_DIR_NAME}/{timestamp}/{BACKORDERS_DATA_FILE_NAME}"
                            self.s3_operations.save_dataframe_as_csv_s3(bucket_name= bucket_name,
                                                                        key= predicted_batch_file_key,
                                                                        dataframe= df)
                            logging.info(f"saved the predicted batch file to s3 at {predicted_batch_file_key}")
                            predicted_batch_timestamps.append(timestamp)
                        else:
                            invalid_batches.append(f"{new_batch_file_key}")
                    if len(predicted_batch_timestamps)>0:
                        logging.info(f"updating the recently predicted batch timestamp")
                        predicted_batch_timestamps.sort(reverse=True)
                        new_recently_predicted_batch_timestamp = predicted_batch_timestamps[0]
                        meta_data_dict = {"recently_predicted_batch_timestamp":new_recently_predicted_batch_timestamp}
                        meta_data = json.dumps(meta_data_dict, indent=2).encode('utf-8')
                        print(meta_data)
                        self.s3_operations.save_object_to_s3(object_body=meta_data, 
                                                            bucket=self.prediction_bucket_name,
                                                            key=self.meta_data_file_key)

                    if len(invalid_batches)>0:
                        logging.info(f"invlaid_batch_Files_found:\n {invalid_batches}")
                else:
                    logging.info(f"all the new batches are already predicted and saved")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)