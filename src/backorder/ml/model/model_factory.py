from backorder.exception import BackorderException
from backorder.logger import logging
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.model_config_file_constants import *
from backorder.utils import read_yaml
import importlib
from typing import List
from collections import namedtuple


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_obj",
                                    "model_serial_number",
                                    "params_grid_search",
                                    "model_name"
                                    ])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", 
                                    ["model_serial_number",
                                        "model",
                                        "best_model",
                                        "best_parameters",
                                        "best_score"])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                    "model",
                                    "best_model",
                                    "best_parameters",
                                    "best_score"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])


class ModelFactory:
    def __init__(self, models_config_file_path: Path = None):
        try:
            self.models_config:dict = read_yaml(models_config_file_path)
            self.grid__search_csv_module = self.models_config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_cv_class = self.models_config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_cv_propertry_data= self.models_config[GRID_SEARCH_KEY][PARAM_KEY]
            self.all_models_intialization_config: dict = self.models_config[MODEL_SELECTION_KEY]

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    @staticmethod
    def get_class_for_name(self,module_name,class_name):
        try:
            module = importlib(module_name)
            class_ref = getattr(module,class_name)
            return class_ref

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def update_parameters_for_model(self,model_obj, params_data):
        try:
            if not isinstance(params_data, dict):
                raise Exception(f"property_data prarameter required to be dictionary")
            for param, value in params_data.items():
                logging.info(f"executing: {str(model_obj)}.{param}= {value}")
                setattr(model_obj,param,value)
            return model_obj

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def get_initialized_model_list(self):
        try:
            initialized_model_list = []
            for model_serial_number in self.all_models_intialization_config.keys():
                single_model_initialization_config = self.all_models_intialization_config[model_serial_number]
                model_class_reference = ModelFactory.get_class_for_name(moduel_name=single_model_initialization_config[MODULE_KEY],
                                                                class_name= single_model_initialization_config[CLASS_KEY])
                model_obj = model_class_reference()

                if PARAM_KEY in single_model_initialization_config:
                    model_obj_params = single_model_initialization_config[PARAM_KEY]
                    initialized_model_obj = self.update_parameters_for_model(model_obj, model_obj_params)
                    
                    grid_params = single_model_initialization_config[SEARCH_PARAM_GRID_KEY]
                    model_name = f"{single_model_initialization_config[MODULE_KEY]}.{single_model_initialization_config[CLASS_KEY]}"
                    initialized_model_list.append(InitializedModelDetail(
                                                        model_name= model_name,
                                                        model_serial_number= model_serial_number,
                                                        model_obj= initialized_model_obj,
                                                        params_grid_search= grid_params
                                                        ))
                self.initialized_model_list =  initialized_model_list
            return initialized_model_list

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
    
    def start_best_parameter_search_for_initialized_model(self,
            initialized_model:InitializedModelDetail,
            input_feature ,
            output_feature):
            try:
                grid_search_class_ref = ModelFactory.get_class_for_name(
                                        module_name=self.grid__search_csv_module,
                                        class_name=self.grid_search_cv_class)

                grid_search_cv_obj = grid_search_class_ref(estimator=initialized_model,
                                      param_grid = initialized_model.params_grid_search)
                self.update_parameters_for_model(model_obj= grid_search_cv_obj,
                                                params_data=self.grid_search_cv_propertry_data)

                grid_search_cv_obj.fit(input_feature, output_feature)
                grid_searched_best_model = GridSearchedBestModel(model= initialized_model,
                                                best_model=grid_search_cv_obj.best_estimator_,
                                                best_score=grid_search_cv_obj.best_score_,
                                                best_parameters=grid_search_cv_obj.best_params_)
                return grid_searched_best_model

            except Exception as e:
                logging.info(BackorderException(e,sys))
                raise BackorderException(e,sys)

    def initiate_best_parameter_search_for_initialized_model(self,
            initialized_model_list: List[InitializedModelDetail],
            input_features,output_features):

            try:
                self.grid_searched_best_model_list= []
                for initialized_model in initialized_model_list:
                    grid_searched_best_model = self.start_best_parameter_search_for_initialized_model(initialized_model= initialized_model,
                                                                                                   input_feature= input_features ,
                                                                                                     output_feature=output_features)
                    self.grid_searched_best_model_list.append(grid_searched_best_model)
                return self.grid_searched_best_model_list

            except Exception as e:
                        logging.info(BackorderException(e,sys))
                        raise BackorderException(e,sys)


    def get_best_model(self,input_x, output_y, base_accuracy = 0.6):
        try:
            logging.info(f"started initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"initialized model list : {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_model(
                                                    initialized_model_list= initialized_model_list,
                                                    input_features = input_x,
                                                    output_features= output_y
                                                    )
            best_model = self.get_best_model_from_grid_searched_best_model_list(
                                                    grid_searched_best_model_list=grid_searched_best_model_list,
                                                    base_accuracy= base_accuracy
                                                    )
            logging.info(f"*********************The model_list is : [{grid_searched_best_model_list}*******************]")
            best_model = ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list =grid_searched_best_model_list,
                                                                                        base_accuracy= base_accuracy)
            return best_model

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

        
    def get_best_model_from_grid_searched_best_model_list(self,grid_searched_best_model_list,
                                                        base_accuracy)-> BestModel: 
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                print(grid_searched_best_model)
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    
                    base_accuracy = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model
            if not best_model:
                    raise Exception(f"None of Model has base accuracy >= {base_accuracy}")
            logging.info(f"Best Model: {best_model}")
            return best_model

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys) from e