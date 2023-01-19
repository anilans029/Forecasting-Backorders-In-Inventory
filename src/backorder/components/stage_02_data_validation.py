from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import DataValidationConfig
from backorder.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.schema_file_constants import *
import pandas as pd
from backorder.utils import read_yaml, write_yaml,create_directories,save_artifacts_to_s3_and_clear_local
from sklearn.model_selection import train_test_split
import shutil
from scipy.stats import ks_2samp
from backorder.data_access import MongodbOperations


class DataValidation:

    def __init__(self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_of_data = read_yaml(SCHEMA_FILE_PATH)
            self.mongo_operations = MongodbOperations()

        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def merged_file_available(self,feature_store_merged_file_path)->bool:
        try:
            merged_file_status = False
            if os.path.exists(feature_store_merged_file_path):
                merged_file_status = True
            logging.info(f"""merged_file_status: {merged_file_status}""")
            return merged_file_status

        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def is_total_no_of_columns_same_as_schema(self,dataframe: pd.DataFrame)->bool:
        try:
            total_no_of_columns = self.schema_of_data[SCHEMA_FILE_TOTAL_NO_COLUMNS]
            total_no_of_num_columns= self.schema_of_data[SCHEMA_FILE_TOTAL_NUMERICAL_COLUMNS]
            total_no_of_cat_columns = self.schema_of_data[SCHEMA_FILE_TOTAL_CATEGORICAL_COLUMNS]
            
            validation_status = True
            total_columns = dataframe.shape[1]
            if total_no_of_columns==total_columns:
                logging.info(f"dataset having same no.of total columns as original schema")
                if total_no_of_num_columns== len(list(dataframe.dtypes.loc[lambda x: x!="O"].index)):
                    logging.info(f"dataset having same no.of total numerical cols as original schema")
                    if total_no_of_cat_columns== len(list(dataframe.dtypes.loc[lambda x: x=="O"].index)):
                        logging.info(f"dataset having same no.of total categorical cols as original schema")
                    else:
                        validation_status= False
                        logging.info(f"""dataset is not having same no.of categorical total columns as original schema
                        dataframe total categorical cols: {len(list(dataframe.dtypes.loc[lambda x: x=="O"].index))} , schema_total_cols: {total_no_of_cat_columns}""")
                else:
                    validation_status = False
                    logging.info(f"""dataset is not having same no.of numerical total columns as original schema
                             dataframe total numerical cols: {len(list(dataframe.dtypes.loc[lambda x: x!="O"].index))} , schema_total_cols: {total_no_of_num_columns}""")
            else:
                validation_status= False
                logging.info(f"dataset is not having same no.of total columns as original schema.\
                             dataframe totatl cols: {total_columns} , schema_total_cols: {total_no_of_columns}")
            return validation_status

        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    @staticmethod
    def get_data(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def check_all_column_names(self, dataframe):
        try:
            missing_cols = []
            for col in self.schema_of_data[SCHEMA_FILE_ALL_COLUMNS]:
                if col in dataframe.columns:
                    continue
                else:
                    missing_cols.append(col)
            if len(missing_cols)==0:
                logging.info(f"all columns are present in the dataset as per schema: {missing_cols}")
                return True
            else:
                logging.info(f"few columns missing in the dataset as per schema: {missing_cols}")
                return False
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    @staticmethod
    def get_num_cat_features(dataframe):
        try:
            num_features= [col for col in dataframe.columns if dataframe[col].dtype!="O"]
            cat_features= [col for col in dataframe.columns if dataframe[col].dtype=="O"]
            return num_features, cat_features
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)


    def is_datatypes_same(self,dataframe: pd.DataFrame):
        try:
            num_feature_datatypes = self.schema_of_data[SCHEMA_FILE_NUMERICAL_FEATURES_DATA_TYPES]
            categorical_datatypes = self.schema_of_data[SCHEMA_FILE_CATEGORICAL_FEATURES_DATA_TYPES]
            
            num_features, cat_features = DataValidation.get_num_cat_features(dataframe=dataframe)
            num_feature_validation_status = True
            cat_feature_validation_status = True
            for col in num_features:
                if not (dataframe[col].dtype == num_feature_datatypes[col]):
                    num_feature_validation_status = False
            logging.info(f"numerical features data types checking completed. Status = {num_feature_validation_status}")
            for col in cat_features:
                if not (dataframe[col].dtype == categorical_datatypes[col]):
                    cat_feature_validation_status = False
            logging.info(f"categorical features data types checking completed. Status = {cat_feature_validation_status}")
            return (num_feature_validation_status and cat_feature_validation_status)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def split_data_to_train_test_validation_sets(self,dataframe):
        try:
            train_set, test_set = train_test_split(dataframe, test_size= self.data_validation_config.test_split_ratio,
                                                    stratify= dataframe[self.schema_of_data["Target_column_Name"]])

            train_set, validation_set = train_test_split(dataframe, test_size= self.data_validation_config.validation_split_ratio,
                                                    stratify= dataframe[self.schema_of_data["Target_column_Name"]])

            return train_set, validation_set, test_set
            # create_directories([self.data_validation_config.ingested_data_dir])
            # logging.info(f"saving the train and test files at: {self.data_validation_config.ingested_data_dir}")
            # train_set.to_csv(self.data_validation_config.train_file_path,index=False)
            # test_set.to_csv(self.data_validation_config.test_file_path,index=False)
            # validation_set.to_csv(self.data_validation_config.validation_file_path,index= False)

        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def check_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame):
        try:
            threshold = self.data_validation_config.ks_2_samp_test_threshold
            drift_report = dict()
            data_drift_found= False
            num_features, col_features = DataValidation.get_num_cat_features(base_df)
            logging.info(f"numerical features: {num_features}")
            for col in num_features:
                logging.info(f"feature: {col}")
                result = ks_2samp(data1= base_df[col], data2= current_df[col], alternative= "less")
                if result.pvalue < threshold:
                    is_drift_found = True
                    data_drift_found = True
                else:
                    is_drift_found = False
                drift_report.update(
                                    {col:{
                                        "p_value": float(result.pvalue),
                                        "is_drift_found": is_drift_found
                                    }}
                )

            drift_file_path = self.data_validation_config.drift_report_file_path
            logging.info(f"saving the drift report at: {drift_file_path}")
            write_yaml(file_path= drift_file_path, content= drift_report)
            return data_drift_found

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def initiate_data_validation(self):
        try:
            logging.info(f"{'*'*10} initiating the data validation {'*'*10}\n")
            feature_store_merged_file_path = self.data_ingestion_artifact.feature_file_path

            #checking if train, test,validation, merged_data files are available or not
            if self.merged_file_available(feature_store_merged_file_path):
                ## loading the merged_data frame
                merged_dataframe = DataValidation.get_data(feature_store_merged_file_path)
                
                total_validation_status = True
                ## checking the raw dataset is having same total no.of columns as per schema file
                validation_status = self.is_total_no_of_columns_same_as_schema(merged_dataframe)
                if not validation_status:
                    total_validation_status= False
                ## checking the raw dataset is having same column names as per schema file
                validation_status = self.check_all_column_names(merged_dataframe)
                if not validation_status:
                    total_validation_status= False
                else:
                    ## checking the data types of all the columns
                    validation_status = self.is_datatypes_same(merged_dataframe)
                    if not validation_status:
                        total_validation_status= False
                    
                ## checking the data drift
                if total_validation_status:
                    logging.info(f"dropping the unwanted features obtained after EDA")
                    unwanted_columns = self.schema_of_data["Unwanted_columns"]
                    merged_dataframe.drop(columns= unwanted_columns,inplace= True)
                    train_set, validation_set, test_set = self.split_data_to_train_test_validation_sets(dataframe= merged_dataframe)

                    logging.info("checking the data_drift.......")
                    status = self.check_data_drift(base_df=train_set,current_df=test_set)
                    
                    create_directories([self.data_validation_config.valid_data_dir])
                    logging.info(f"saving the train, test, valid files at: {self.data_validation_config.valid_data_dir}")
                    train_set.to_csv(self.data_validation_config.valid_train_file_path,index=False)
                    test_set.to_csv(self.data_validation_config.valid_test_file_path,index=False)
                    validation_set.to_csv(self.data_validation_config.valid_validation_file_path,index= False)
                    data_validation_artifact = DataValidationArtifact(
                                                    training_phase= "Data_validation",
                                                    validation_status= total_validation_status,
                                                    valid_train_filepath= self.data_validation_config.valid_train_file_path,
                                                    valid_test_filepath=self.data_validation_config.valid_test_file_path,
                                                    valid_validation_file_path= self.data_validation_config.valid_validation_file_path,
                                                    Invalid_filepath= None,
                                                    drift_report_filepath=self.data_validation_config.drift_report_file_path
                                                    ) 

                    data_validation_artifact_dict = data_validation_artifact.__dict__
                    data_validation_artifact_dict["valid_train_filepath"] = str(data_validation_artifact_dict['valid_train_filepath'])
                    data_validation_artifact_dict["valid_test_filepath"] = str(data_validation_artifact_dict['valid_test_filepath'])
                    data_validation_artifact_dict["valid_validation_file_path"] = str(data_validation_artifact_dict['valid_validation_file_path'])
                    data_validation_artifact_dict["drift_report_filepath"] = str(data_validation_artifact_dict['drift_report_filepath'])
                 
                    self.mongo_operations.save_artifact(artifact= data_validation_artifact_dict)
                    logging.info(f"saved the data_validation artifact to mongodb")

                    logging.info(f"data_validation artifact: {data_validation_artifact}")
                    logging.info(f"{'*'*10} completed the data validation {'*'*10}\n")
                    return data_validation_artifact
                else:
                    logging.info("saving the data into invalid data dir")
                    create_directories([self.data_validation_config.invalid_data_dir])
                    merged_dataframe.to_csv(self.data_validation_config.invalid_file_path,index= False)    
                    data_validation_artifact = DataValidationArtifact(
                                                    training_phase= "Data_validation",
                                                    validation_status= total_validation_status,
                                                    valid_train_filepath= None,
                                                    valid_test_filepath=  None,
                                                    valid_validation_file_path= None,
                                                    Invalid_filepath= self.data_validation_config.invalid_file_path,
                                                    drift_report_filepath=None
                                                    )   
                    data_validation_artifact_dict = data_validation_artifact.__dict__
                    data_validation_artifact_dict["Invalid_filepath"] = str(data_validation_artifact_dict['Invalid_filepath'])
                    data_validation_artifact_dict["drift_report_filepath"] = str(data_validation_artifact_dict['drift_report_filepath'])
                    self.mongo_operations.save_artifact(artifact= data_validation_artifact_dict)
                    logging.info(f"saved the data_validation artifact to mongodb")
                    
                    logging.info(f"data_validation artifact: {data_validation_artifact}")
                    logging.info(f"{'*'*10} completed the data validation {'*'*10}\n")
                    return data_validation_artifact

                
            else:
                raise Exception(f"Merged data file is not available in the ingested data folder")

        except Exception as e:
            save_artifacts_to_s3_and_clear_local()
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
