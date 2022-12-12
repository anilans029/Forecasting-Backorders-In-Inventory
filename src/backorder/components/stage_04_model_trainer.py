from backorder.exception import BackorderException
from backorder.logger import logging
from backorder.entity.config_entity import ModelTrainerConfig
from backorder.entity.artifact_entity import ModelTrainerArtiact, DataTransformationArtifact
from backorder.config.pipeline.training_configuration_manager import TrainingConfigurationManager
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.schema_file_constants import *
import pandas as pd
from backorder.utils import read_yaml,create_directories
import shutil
import pandas as pd
import numpy as np
from backorder.utils import load_numpy_array_data, load_object,save_object
from backorder.ml import Mo


class ModelTrainer:

    def __init__(self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def initiate_model_training(self):
        try:
            logging.info(f"{'*'*10} initiating the Model Training {'*'*10}\n")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            validation_file_path = self.data_transformation_artifact.transformed_valid_file_path

            logging.info(f"loading the transformed train,validation data")
            train_arr = load_numpy_array_data(train_file_path)
            validation_arr = load_numpy_array_data(validation_file_path)

            logging.info(f"seperating the independent and dependent features in train arr")
            train_x_arr = train_arr[:,:-1]
            train_y_arr = train_arr[:, -1]

            logging.info(f"seperating the independent and dependent features in validation arr")
            validation_x_arr = validation_arr[:,:-1]
            validation_y_arr = validation_arr[:,-1]

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)