from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import DataTransformationConfig
from backorder.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.schema_file_constants import *
import pandas as pd
from backorder.utils import read_yaml, write_yaml,create_directories
import shutil
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from backorder.ml.model.esitmator import TargetValueMapping,FeatureScaler,NullImputer,DataPreprocessor
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from backorder.utils import save_numpy_array_data, save_object
from backorder.data_access import MongodbOperations


class DataTransformation:

    def __init__(self, data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig):

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_of_data = read_yaml(SCHEMA_FILE_PATH)
            self.mongo_operations = MongodbOperations()
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)
    
    def get_data(self,filepath):
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                return df
            else:
                raise Exception(f"file doesn't exists at {filepath}")

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def get_num_cat_features(self,dataframe: pd.DataFrame):
        try:
            num_features= [col for col in dataframe.columns if dataframe[col].dtype!="O"]
            cat_features= [col for col in dataframe.columns if dataframe[col].dtype=="O"]
            return num_features, cat_features
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def get_num_Nan_imputer(self,numerical_features):
        try:
            numerical_Nan_imputer = ColumnTransformer([
                ("SimpleImputer_with_median",SimpleImputer(strategy="median"),numerical_features)
            ], remainder='passthrough')
            return numerical_Nan_imputer

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
    


        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def get_preprocessor(self,num_features,cat_features):
        try:
            numeric_transformer = Pipeline(steps = [
                                                    ("missing_value_imputer", SimpleImputer(strategy="median")),
                                                    ("feature_scaling", RobustScaler())
                                                    ])

            categorical_transformer = Pipeline(steps= [
                                                    ("categorical_encoding",OneHotEncoder(handle_unknown="ignore", drop="first"))
                                                    ])


            preprocessor = ColumnTransformer(transformers= [
                                ("cat_transformer", categorical_transformer, cat_features),
                                ("num_transformer", numeric_transformer,num_features),
            ],remainder='passthrough'
            )
            
            # preprocessor = ColumnTransformer([
            #     ("numerical_transformer",FeatureScaler(),num_features),
            #     ("categorical_encoder",OneHotEncoder(handle_unknown='ignore',drop='first'),cat_features)
            # ],remainder='passthrough')
            return preprocessor

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def initiate_data_transformation(self):
        try:
            if self.data_validation_artifact.validation_status:
                logging.info(f"{'*'*10} initiating the data Transformation {'*'*10}\n")
                train_file_path = self.data_validation_artifact.valid_train_filepath
                test_file_path = self.data_validation_artifact.valid_test_filepath
                valid_file_path = self.data_validation_artifact.valid_validation_file_path
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
                transformed_valid_file_path = self.data_transformation_config.transformed_valid_file_path
                preprocesed_obj_file_path = self.data_transformation_config.preprocessor_obj_file_path

                logging.info(f"loading the valid train, test, valid dataset files from valid data folder")
                train_df = self.get_data(train_file_path)
                test_df = self.get_data(test_file_path)
                valid_df = self.get_data(valid_file_path)

                logging.info(f"seperating the independent and dependent features for train df")
                independent_train_df = train_df.drop(columns=[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]])
                target_feature_train_df = train_df[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]].replace(TargetValueMapping().to_dict())

                logging.info(f"seperating the independent and dependent features for test df")
                independent_test_df = test_df.drop(columns=[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]])
                target_feature_test_df = test_df[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]].replace(TargetValueMapping().to_dict())

                logging.info(f"seperating the independent and dependent features for valid df")
                independent_valid_df = valid_df.drop(columns=[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]])
                target_feature_valid_df = valid_df[self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]].replace(TargetValueMapping().to_dict())
                
                ### numerical_features, categorical_features, target_feature
                num_features, cat_features = self.get_num_cat_features(dataframe= train_df)
                target_feature = self.schema_of_data[SCHEMA_FILE_TARGET_COLUMN_NAME]
                cat_features.remove(target_feature)
            
                #### Commented the preprocessor steps after getting better results in latest analysis
                # logging.info(f"imputing the null values for the training set")
                # nan_imputer = NullImputer(num_features_list=num_features)
                # transformed_train_df = nan_imputer.fit_transform(independent_train_df)

                # logging.info(f"imputing the null values for the testing set")
                # transformed_test_df = nan_imputer.transform(independent_test_df)

                # logging.info(f"imputing the null values for the validation set")
                # transformed_validation_df = nan_imputer.transform(independent_valid_df)

                # logging.info(f"handling the imbalance data in the train dataset....")
                # cat_features_index= [independent_train_df.columns.get_loc(col) for col in independent_train_df.columns if independent_train_df[col].dtype=="O"]
                # smotenc = SMOTENC(categorical_features=cat_features_index)
                # train_independent_res,train_target_res =  smotenc.fit_resample(X= transformed_train_df, y= target_feature_train_df)
                
                logging.info("getting the preprocessor and performing the transformations")
                preprocessor = self.get_preprocessor(num_features,cat_features)
                transf_train_feature_arr = preprocessor.fit_transform(independent_train_df)
                transf_test_feature_arr = preprocessor.transform(independent_test_df)
                transf_valid_feature_arr = preprocessor.transform(independent_valid_df)

                logging.info("combining both the input and target features into single array for both train and test")
                train_arr = np.c_[transf_train_feature_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[transf_test_feature_arr, np.array(target_feature_test_df)]
                valid_arr = np.c_[transf_valid_feature_arr, np.array(target_feature_valid_df)]

                ### saving the test,train,valid data arrays in their respective paths
                logging.info(f"""Saving the preprocessed train file at {[transformed_train_file_path]}
                                test data files at {[transformed_test_file_path]}
                                validation data file at {[transformed_valid_file_path]}.""")
                save_numpy_array_data(transformed_train_file_path,train_arr)
                save_numpy_array_data(transformed_test_file_path, test_arr)
                save_numpy_array_data(transformed_valid_file_path, valid_arr)

                logging.info(f"saving the preprocessed obj at {preprocesed_obj_file_path} ")
                save_object(file_path=preprocesed_obj_file_path, obj=preprocessor)

                data_transformation_artifact = DataTransformationArtifact(
                                        training_phase= "Data_Transformation",
                                        is_Transformed = True ,
                                        transformed_train_file_path= transformed_train_file_path ,
                                        transformed_test_file_path= transformed_test_file_path,
                                        transformed_obj_file_path= preprocesed_obj_file_path,
                                        transformed_valid_file_path=transformed_valid_file_path,
                                        message= "Data Transformation completed succesfully")
                
                logging.info(f"data_Transformation artifact: {data_transformation_artifact}")
                data_transformation_artifact_dict = data_transformation_artifact.__dict__
                data_transformation_artifact_dict["transformed_train_file_path"] = str(data_transformation_artifact_dict['transformed_train_file_path'])
                data_transformation_artifact_dict["transformed_test_file_path"] = str(data_transformation_artifact_dict['transformed_test_file_path'])
                data_transformation_artifact_dict["transformed_obj_file_path"] = str(data_transformation_artifact_dict['transformed_obj_file_path'])
                data_transformation_artifact_dict["transformed_valid_file_path"] = str(data_transformation_artifact_dict['transformed_valid_file_path'])
                self.mongo_operations.save_artifact(artifact= data_transformation_artifact_dict)
                logging.info(f"saved the data_transformation artifact to mongodb")

                logging.info(f"{'*'*10} completed the data Transformation {'*'*10}\n")
                return data_transformation_artifact
            else:
                logging.info(f"since validation status is false, not initiating the data transformation phase ")
                raise Exception(f"since validation status is false, not initiating the data transformation phase")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
