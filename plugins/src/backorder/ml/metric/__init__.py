from backorder.exception import BackorderException
from backorder.logger import logging
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.model_config_file_constants import *
import numpy as np
from sklearn.metrics import f1_score,precision_score, recall_score,roc_auc_score
from collections import namedtuple
from backorder.utils import save_object


MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_recall", "test_recall", "train_precision",
                                 "test_precision", "train_F1", "test_F1","model_accuracy"])

class EvaluateClassificationModel:
    def __init__(self,train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        base_accuracy: float = 0.6):
        try:
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
            self.base_accuracy = base_accuracy
        except Exception as e:
            logging.info(BaseException((e,sys)))
            raise BackorderException(e,sys)
        
    def evaluate_classification_model(self,model_list):
        try:
            metric_info_artifact = None
            for model in model_list:
                ## making the predicitions for the train and test sets
                train_y_pred = model.predict(self.train_x)
                test_y_pred = model.predict(self.test_x)
               
                ## calculating the f1_score for both train and test sets
                train_f1_score = f1_score(self.train_y, train_y_pred)
                test_f1_score = f1_score(self.test_y, test_y_pred)

                ## calculating the precision_score for both train and test sets
                train_precision_score = precision_score(self.train_y, train_y_pred)
                test_precision_score = precision_score(self.test_y, test_y_pred)

                ## calculating the recall_score for both train and test sets
                train_recall_score = recall_score(self.train_y, train_y_pred)
                test_recall_score = recall_score(self.test_y, test_y_pred)

                ## calculating the model_accuracy by harmonic mean of trian_f1_score and test_f1_score
                model_accuracy = (2*(train_f1_score* test_f1_score))/(train_f1_score + test_f1_score)
                diff_test_train_acc = abs(train_f1_score - test_f1_score)

                #logging all important metrics
                logging.info(f"\n\nClassificatin metrics for the model '{model.__class__}': ")
                logging.info(f"{'**'* 15} F1-Score {'**'* 15}")
                logging.info(f"Train Score\t\t Test Score\t\t Average Score")
                logging.info(f"{train_f1_score}\t\t {test_f1_score}\t\t{model_accuracy}")

                logging.info(f"{'**'* 15} precision_score {'**'* 15}")
                logging.info(f"Train Score\t\t Test Score")
                logging.info(f"{train_precision_score}\t\t {train_precision_score}")

                logging.info(f"{'**'* 15} recall_score {'**'* 15}")
                logging.info(f"Train Score\t\t Test Score\t\t Average Score")
                logging.info(f"{train_recall_score}\t\t {test_recall_score}")

                logging.info(f"Diff of  test, train accuracy: [{diff_test_train_acc}].") 
                logging.info(f"Checking if the model is having better accuracy")
                if (model_accuracy > self.base_accuracy) and (diff_test_train_acc <= 0.9):
                    self.base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name= str(model),
                                                            model_object=model,
                                                            train_recall= train_recall_score,
                                                            test_recall=test_recall_score,
                                                            train_precision= train_precision_score,
                                                            test_precision= test_precision_score,
                                                            train_F1= train_f1_score,
                                                            test_F1= test_f1_score,
                                                            model_accuracy= model_accuracy
                                                            )
                    logging.info(f"Acceptable model found {metric_info_artifact}. ")
                else:
                    logging.info(f"This model is not having better accuracy")
            if metric_info_artifact is None:
                raise Exception(f"No model is satifying the base accuracy, hence decling all the models")
            return metric_info_artifact
    
        except Exception as e:
            logging.info(BaseException((e,sys)))
            raise BackorderException(e,sys)
    

    def evaluate_and_get_classification_metric(self,model):
        try:
            logging.info(f"started evaluating the {model.__class__} object :")
            ## making the predicitions for the train and test sets
            train_y_pred = model.predict(self.train_x)
            test_y_pred = model.predict(self.test_x)
            
            ## calculating the f1_score for both train and test sets
            train_f1_score = f1_score(self.train_y, train_y_pred)
            test_f1_score = f1_score(self.test_y, test_y_pred)

            ## calculating the precision_score for both train and test sets
            train_precision_score = precision_score(self.train_y, train_y_pred)
            test_precision_score = precision_score(self.test_y, test_y_pred)

            ## calculating the recall_score for both train and test sets
            train_recall_score = recall_score(self.train_y, train_y_pred)
            test_recall_score = recall_score(self.test_y, test_y_pred)

            ## calculating the model_accuracy by harmonic mean of trian_f1_score and test_f1_score
            model_accuracy = (2*(train_f1_score* test_f1_score))/(train_f1_score + test_f1_score)
            diff_test_train_acc = abs(train_f1_score - test_f1_score)

            #logging all important metrics
            logging.info(f"\n\nClassificatin metrics for the model '{model.__class__}': ")
            logging.info(f"{'**'* 15} F1-Score {'**'* 15}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_f1_score}\t\t {test_f1_score}\t\t{model_accuracy}")

            logging.info(f"{'**'* 15} precision_score {'**'* 15}")
            logging.info(f"Train Score\t\t Test Score")
            logging.info(f"{train_precision_score}\t\t {train_precision_score}")

            logging.info(f"{'**'* 15} recall_score {'**'* 15}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_recall_score}\t\t {test_recall_score}")

            logging.info(f"Diff of  test, train accuracy: [{diff_test_train_acc}].")

            metric_info_artifact = MetricInfoArtifact(model_name= str(model.__class__),
                                                            model_object=model,
                                                            train_recall= train_recall_score,
                                                            test_recall=test_recall_score,
                                                            train_precision= train_precision_score,
                                                            test_precision= test_precision_score,
                                                            train_F1= train_f1_score,
                                                            test_F1= test_f1_score,
                                                            model_accuracy= model_accuracy
                                                            )
            return metric_info_artifact
        except Exception as e:
            logging.info(BaseException((e,sys)))
            raise BackorderException(e,sys)