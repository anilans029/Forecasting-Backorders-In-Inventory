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
from backorder.data_access import MongodbOperations

class ModelTrainer:

    def __init__(self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.mongo_operations = MongodbOperations()

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def initiate_model_training(self):
        try:
            if self.data_transformation_artifact.is_Transformed:
                logging.info(f"{'*'*10} initiating the Model Training {'*'*10}\n")
                trained_model_file_path = self.model_trainer_config.trained_model_file_path
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
                logging.info(f"best model found on training dataset: {best_model.best_model.__class__.__name__}")

                logging.info(f"now evaluating all the trianed models on both trained and validation sets")
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
                                pr_auc: {metric_info_artifact.pr_auc_test}
                                roc_auc: {metric_info_artifact.test_roc_auc}""")
                
                ## loading the transformer object
                transformer_obj = load_object(self.data_transformation_artifact.transformed_obj_file_path)

                backorder_model_obj = BackoderModel(transformer_obj= transformer_obj, 
                                                    model_obj= metric_info_artifact.model_object)
                
                logging.info(f"saving the trained model object at : {trained_model_file_path}")
                save_object(file_path=trained_model_file_path,
                            obj= backorder_model_obj)

                model_trainer_artifact = ModelTrainerArtiact(
                                                            training_phase= "Model_Training",
                                                            trained_model_file_path= trained_model_file_path,
                                                            is_model_found= True,
                                                            train_recall= metric_info_artifact.train_recall,
                                                            test_recall=metric_info_artifact.test_recall,
                                                            train_f1_each_class_scores=metric_info_artifact.train_f1_each_class_scores,
                                                            test_f1_each_class_scores=metric_info_artifact.test_f1_each_class_scores,
                                                            train_precision= metric_info_artifact.train_precision,
                                                            test_precision= metric_info_artifact.test_precision,
                                                            train_macro_F1= metric_info_artifact.train_macro_F1,
                                                            test_macro_F1= metric_info_artifact.test_macro_F1,
                                                            train_roc_auc=metric_info_artifact.train_roc_auc,
                                                            test_roc_auc=metric_info_artifact.test_roc_auc,
                                                            pr_auc_train=metric_info_artifact.pr_auc_train,
                                                            pr_auc_test=metric_info_artifact.pr_auc_test)

                model_trainer_artifact_dict = model_trainer_artifact.__dict__
                model_trainer_artifact_dict["trained_model_file_path"] = str(model_trainer_artifact_dict['trained_model_file_path'])
                model_trainer_artifact_dict["train_f1_each_class_scores"] = list(model_trainer_artifact_dict['train_f1_each_class_scores'])
                model_trainer_artifact_dict["test_f1_each_class_scores"] = list(model_trainer_artifact_dict['test_f1_each_class_scores'])

                self.mongo_operations.save_artifact(artifact= model_trainer_artifact_dict)
                logging.info(f"saved the model_trainer artifact to mongodb")

                logging.info(f"model_trainer_artifact = {model_trainer_artifact}")
                logging.info(f"{'*'*10} completed the Model Training {'*'*10}\n\n")
                return model_trainer_artifact
            else:
                raise Exception(f"since Data_Transformations_status is False, not initiating the Model Training Phase")
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)