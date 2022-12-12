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
from backorder.ml.model.model_factory import ModelFactory
from backorder.ml.metric import EvaluateClassificationModel, MetricInfoArtifact
from backorder.ml.model.esitmator import BackoderModel


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
            trained_model_file_path = self.model_trainer_config.trained_model_file_path


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

            ## initializing the model factory
            model_factory = ModelFactory(models_config_file_path= self.model_trainer_config.models_config_file_path)
            
            logging.info(f'getting the best model using model factory')
            best_model = model_factory.get_best_model(input_x=train_x_arr,
                                                    output_y=train_y_arr,
                                                    base_accuracy= self.model_trainer_config.base_accuracy)
            logging.info(f"best model found on training dataset: {best_model}")

            logging.info(f"now evaluating the all the trianed models on both trained and validation sets")
            grid_searched_best_model_list = model_factory.grid_searched_best_model_list
            model_list = [model.best_model for model in grid_searched_best_model_list]
            evaluate_classification_model = EvaluateClassificationModel(base_accuracy= self.model_trainer_config.base_accuracy,
                                                                            train_x= train_x_arr,
                                                                            train_y= train_y_arr,
                                                                            test_x= validation_x_arr,
                                                                            test_y= validation_y_arr 
                                                                            )
            metric_info_artifact: MetricInfoArtifact = evaluate_classification_model.evaluate_classification_model(model_list)
            logging.info(f"""best model found after evaluating on training and validation sets: 
                             model:{metric_info_artifact.model_name}
                             accuracy: {metric_info_artifact.model_accuracy}""")
            
            ## loading the transformer object
            transformer_obj = load_object(self.data_transformation_artifact.transformed_obj_file_path)

            backorder_model_obj = BackoderModel(transformer_obj= transformer_obj, 
                                                model_obj= metric_info_artifact.model_object)
            
            logging.info(f"saving the trained model object at : {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,
                        obj= metric_info_artifact.model_object)

            model_trainer_artifact = ModelTrainerArtiact(trained_model_file_path= trained_model_file_path,
                                                        is_model_found= True,
                                                        train_f1_score= metric_info_artifact.train_F1,
                                                        validation_f1_score=metric_info_artifact.test_F1,
                                                        train_precision= metric_info_artifact.train_precision,
                                                        train_recall= metric_info_artifact.train_recall,
                                                        validation_precision= metric_info_artifact.test_precision,
                                                        validation_recall= metric_info_artifact.test_recall)
            logging.info(f"model_trainer_artifact = {model_trainer_artifact}")
            logging.info(f"{'*'*10} completed the Model Training {'*'*10}\n\n")
            return model_trainer_artifact
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)