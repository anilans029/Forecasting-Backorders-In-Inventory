from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import DataValidationConfig
from backorder.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.schema_file_constants import *
import pandas as pd
from backorder.utils import read_yaml, write_yaml,create_directories
import shutil
from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_of_data = read_yaml(SCHEMA_FILE_PATH)
            
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def is_train_test_merged_files_available(self, train_file_path, test_file_path,
                        feature_store_merged_file_path,validation_file_path)->bool:
        try:
            train_file_status, test_file_status,merged_file_status = False, False,False
            validation_file_status = False
            if os.path.exists(train_file_path):
                train_file_status = True
            if os.path.exists(test_file_path):
                test_file_status = True
            if os.path.exists(feature_store_merged_file_path):
                merged_file_status = True
            if os.path.exists(validation_file_path):
                validation_file_status = True
            logging.info(f"""train_file_available_status: {train_file_status}
                             test_file_available_status: {test_file_status}
                             merged_file_status: {merged_file_status}
                             validation_file_status: {validation_file_status}""")
            is_train_test_files_available = (train_file_status and test_file_status)and (merged_file_status and validation_file_status)
            return is_train_test_files_available

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
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            validation_file_path = self.data_ingestion_artifact.validation_file_path
            feature_store_merged_file_path = self.data_ingestion_artifact.feature_file_path

            #checking if train, test,validation, merged_data files are available or not
            if self.is_train_test_merged_files_available(train_file_path, test_file_path,feature_store_merged_file_path,validation_file_path):
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
                    train_df = DataValidation.get_data(train_file_path)
                    test_df = DataValidation.get_data(test_file_path)
                    logging.info("checking the data_drift.......")
                    status = self.check_data_drift(base_df=train_df,current_df=test_df)
                    
                    logging.info("moving the train and test dataset to valid data dir")
                    create_directories([self.data_validation_config.valid_data_dir])
                    shutil.copy(self.data_ingestion_artifact.train_file_path, self.data_validation_config.valid_train_file_path)
                    shutil.copy(self.data_ingestion_artifact.test_file_path, self.data_validation_config.valid_test_file_path)
                    shutil.copy(self.data_ingestion_artifact.validation_file_path, self.data_validation_config.valid_validation_file_path)
                    data_validation_artifact = DataValidationArtifact(
                                                    validation_status= total_validation_status,
                                                    valid_train_filepath= self.data_validation_config.valid_train_file_path,
                                                    valid_test_filepath=self.data_validation_config.valid_test_file_path,
                                                    valid_validation_file_path= self.data_validation_config.valid_validation_file_path,
                                                    Invalid_test_filepath= None,
                                                    Invalid_train_filepath=None,
                                                    Invalid_validation_file_path=None,
                                                    drift_report_filepath=self.data_validation_config.drift_report_file_path
                                                    )   
                    logging.info(f"data_validation artifact: {data_validation_artifact}")
                    logging.info(f"{'*'*10} completed the data validation {'*'*10}\n")
                    return data_validation_artifact
                else:
                    logging.info("moving the train and test dataset to invalid data dir")
                    create_directories([self.data_validation_config.invalid_data_dir])
                    shutil.copy(self.data_ingestion_artifact.train_file_path, self.data_validation_config.invalid_train_file_path)
                    shutil.copy(self.data_ingestion_artifact.test_file_path,self.data_validation_config.invalid_test_file_path)
                    shutil.copy(self.data_ingestion_artifact.validation_file_path, self.data_validation_config.invalid_validation_file_path)    
                    data_validation_artifact = DataValidationArtifact(
                                                    validation_status= total_validation_status,
                                                    valid_train_filepath= None,
                                                    valid_test_filepath=  None,
                                                    valid_validation_file_path= None,
                                                    Invalid_test_filepath= self.data_validation_config.invalid_test_file_path,
                                                    Invalid_train_filepath=self.data_validation_config.invalid_train_file_path,
                                                    Invalid_validation_file_path=self.data_validation_config.invalid_validation_file_path,
                                                    drift_report_filepath=None
                                                    )   
                    logging.info(f"data_validation artifact: {data_validation_artifact}")
                    logging.info(f"{'*'*10} completed the data validation {'*'*10}\n")
                    return data_validation_artifact

                
            else:
                raise Exception(f"Train or test files is not available in the ingested data folder")

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
