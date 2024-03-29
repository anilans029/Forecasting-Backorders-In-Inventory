from backorder.exception import BackorderException
from backorder.logger import logging
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.model_config_file_constants import *
from backorder.utils import read_yaml
import importlib
from typing import List
from sklearn.metrics import auc, precision_recall_curve, make_scorer
from collections import namedtuple
import numpy as np
from sklearn.metrics import auc


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
            self.randomized_grid__search_cv_module = self.models_config[GRID_SEARCH_KEY][MODULE_KEY]
            self.randomized_grid_search_cv_class = self.models_config[GRID_SEARCH_KEY][CLASS_KEY]
            self.randomized_grid_search_cv_propertry_data= self.models_config[GRID_SEARCH_KEY][PARAM_KEY]
            self.all_models_intialization_config: dict = self.models_config[MODEL_SELECTION_KEY]

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    @staticmethod
    def get_class_for_name(module_name,class_name):
        try:
            logging.info(f"getting class :{class_name} from the module: {module_name}")
            module = importlib.import_module(module_name)
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
                logging.info(f"updating prams for {model_obj.__class__.__name__} object: {param}= {value}")
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
                model_class_reference = ModelFactory.get_class_for_name(module_name=single_model_initialization_config[MODULE_KEY],
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
                                                    model_obj= initialized_model_obj if PARAM_KEY in single_model_initialization_config else model_obj,
                                                    params_grid_search= grid_params
                                                    ))
                self.initialized_model_list =  initialized_model_list
            return initialized_model_list

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
    
    # @staticmethod
    # def pr_auc_score(y_true, y_score):
    #     """
    #     Generates the Area Under the Curve for precision and recall.
    #     """
    #     precision, recall, thresholds = precision_recall_curve(y_true, y_score[:, 1])
    #     return auc(recall, precision, reorder=True)

    def start_best_parameter_search_for_initialized_model(self,
            initialized_model:InitializedModelDetail,
            input_feature ,
            output_feature):
            try:
                randomized_search_class_ref = ModelFactory.get_class_for_name(
                                        module_name=self.randomized_grid__search_cv_module,
                                        class_name=self.randomized_grid_search_cv_class)

                randomized_search_cv_obj = randomized_search_class_ref(estimator=initialized_model.model_obj,
                                      param_distributions = initialized_model.params_grid_search)
                randomized_searchcv_params = self.randomized_grid_search_cv_propertry_data
                randomized_searchcv_params["scoring"]= "roc_auc"
                
                self.update_parameters_for_model(model_obj= randomized_search_cv_obj,
                                                params_data=randomized_searchcv_params)
                logging.info(f"started searching the best params for the model: {initialized_model.model_name}")
                randomized_search_cv_obj.fit(input_feature, output_feature)
                logging.info(f"best params found for the {initialized_model.model_name} are : {randomized_search_cv_obj.best_params_}")
                grid_searched_best_model = GridSearchedBestModel(model= initialized_model,
                                                best_model=randomized_search_cv_obj.best_estimator_,
                                                model_serial_number = initialized_model.model_serial_number,
                                                best_score=randomized_search_cv_obj.best_score_,
                                                best_parameters=randomized_search_cv_obj.best_params_)
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
                    logging.info(f"""Grid_searched_best_model:
                                    model = {grid_searched_best_model.best_model.__class__.__name__}
                                    model_serial_number = {grid_searched_best_model.model_serial_number}
                                    best_parameters= {grid_searched_best_model.best_parameters}""")
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
            
            logging.info(f"\nBest models list after do grid search: {[model.best_model.__class__.__name__ for model in grid_searched_best_model_list]}")
            logging.info(f"Now finding the best model out of all the models in the above list")
            best_model = self.get_best_model_from_grid_searched_best_model_list(
                                                    grid_searched_best_model_list=grid_searched_best_model_list,
                                                    base_accuracy= base_accuracy,
                                                    input_x = input_x,
                                                    output_y = output_y
                                                    )
            return best_model

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

        
    def get_best_model_from_grid_searched_best_model_list(self,grid_searched_best_model_list,
                                                               base_accuracy,
                                                               input_x,
                                                               output_y)-> BestModel: 
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                y_pred_prob =  grid_searched_best_model.best_model.predict_proba(input_x)[:,1]
                precision, recall,_ =  precision_recall_curve(output_y, y_pred_prob)
                pr_auc_score = np.round(auc(recall, precision),4)
                if base_accuracy < pr_auc_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model.best_model.__class__.__name__} with pr_auc_score = {pr_auc_score}")
                    base_accuracy = pr_auc_score
                    best_model = grid_searched_best_model
            if not best_model:
                    raise Exception(f"None of Model has base accuracy >= {base_accuracy}")
            logging.info(f"Best Model = {best_model.best_model.__class__.__name__}")
            return best_model

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys) from e