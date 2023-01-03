from backorder.exception import BackorderException
from backorder.logger import logging
import os, sys
from pathlib import Path
from backorder.constants.training_pipeline_config.model_config_file_constants import *
import numpy as np
from sklearn.metrics import f1_score,precision_score, recall_score,roc_auc_score, roc_curve,precision_recall_curve, auc
from collections import namedtuple
from backorder.utils import save_object
from prettytable import PrettyTable
from dataclasses import dataclass

@dataclass(frozen=True)
class MetricInfoArtifact:
    model_name:str
    model_object:any
    train_recall:float
    train_f1_each_class_scores:np.array
    test_f1_each_class_scores:np.array
    test_recall:float
    train_precision:float
    test_precision:float
    train_macro_F1:float
    test_macro_F1:float
    train_roc_auc:float
    test_roc_auc:float
    pr_auc_train:float
    pr_auc_test:float

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

    def find_best_threshold(self,threshold, fpr, tpr):
        t = threshold[np.argmax(tpr*(1-fpr))]
        # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
        return t

    def predict_with_best_t(self,proba, threshold):
        predictions = []
        for i in proba:
            if i>=threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
        
    def evaluate_classification_model(self,model_list):
        try:
            metric_info_artifact = None
            for model in model_list:
                ## making the predicitions for the train and test sets
                train_y_pred_prob = model.predict_proba(self.train_x)[:,1]
                test_y_pred_prob = model.predict_proba(self.test_x)[:,1]
                model_name = model.__class__.__name__

                train_roc_auc = np.round(roc_auc_score(self.train_y, train_y_pred_prob),4) #Calculate Roc_auc score
                test_roc_auc = np.round(roc_auc_score(self.test_y, test_y_pred_prob),4) #Calculate Roc_auc score
                
                train_fpr,train_tpr,train_threshold=roc_curve(self.train_y,train_y_pred_prob)
                precision_train,recall_train,_ =  precision_recall_curve(self.train_y, train_y_pred_prob) 
                precision_test,recall_test,_ = precision_recall_curve(self.test_y, test_y_pred_prob)
                
                pr_auc_train_score = np.round(auc(recall_train, precision_train),4)
                pr_auc_test_score =  np.round(auc(recall_test, precision_test),4)

                best_thres = self.find_best_threshold(threshold= train_threshold,fpr = train_fpr, tpr = train_tpr)
                train_pred = self.predict_with_best_t(train_y_pred_prob, best_thres)
                test_pred = self.predict_with_best_t(test_y_pred_prob, best_thres)
                
                # for training
                train_f1_macro = np.round(f1_score(self.train_y, train_pred, average = 'macro'),4) # calculating macro f1 score
                train_f1_each_class_scores = np.round(f1_score(self.train_y, train_pred, average = None  ),4) # calculating f1 score for each class
                train_precision = np.round(precision_score(self.train_y, train_pred),4) # Calculate Precision
                train_recall = np.round(recall_score(self.train_y, train_pred),4)  # Calculate Recall

                #for testing
                test_f1_macro = np.round(f1_score(self.test_y, test_pred, average = 'macro'),4) #calculate the macro_f1 score
                test_f1_each_class_scores = np.round(f1_score(self.test_y, test_pred, average = None),4) # calculating f1 score for each class
                test_precision = np.round(precision_score(self.test_y, test_pred),4) # Calculate Precision
                test_recall = np.round(recall_score(self.test_y, test_pred),4)  # Calculate Recall
                logging.info(f"\n\nClassificatin metrics for the model '{model.__class__.__name__}': ")
                table = PrettyTable(["set","macro_f1_score","each_class_f1_score","precision_score","recall_score","roc_auc_score","pr_auc_score"])
                table.add_row(["train",train_f1_macro,train_f1_each_class_scores, train_precision,train_recall,train_roc_auc,pr_auc_train_score])
                table.add_row(["test",test_f1_macro,test_f1_each_class_scores, train_precision,test_recall,test_roc_auc,pr_auc_test_score])
                logging.info(f"\n\n{table}")


                model_pr_auc = pr_auc_test_score
                diff_test_train_roc_auc= abs(train_roc_auc - test_roc_auc)

                logging.info(f"Diff of  test, train roc_auc: [{diff_test_train_roc_auc}].") 
                logging.info(f"Checking if the model is having better pr_auc")
                if (model_pr_auc > self.base_accuracy) and (diff_test_train_roc_auc <= 0.8):
                    self.base_accuracy = model_pr_auc
                    metric_info_artifact = MetricInfoArtifact(model_name= model_name,
                                                            model_object=model,
                                                            train_recall= train_recall,
                                                            test_recall=test_recall,
                                                            train_f1_each_class_scores=train_f1_each_class_scores,
                                                            test_f1_each_class_scores=test_f1_each_class_scores,
                                                            train_precision= train_precision,
                                                            test_precision= test_precision,
                                                            train_macro_F1= train_f1_macro,
                                                            test_macro_F1= test_f1_macro,
                                                            train_roc_auc=train_roc_auc,
                                                            test_roc_auc=test_roc_auc,
                                                            pr_auc_train=pr_auc_train_score,
                                                            pr_auc_test=pr_auc_test_score)
                                                            
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
            ## making the predicitions for the train and test sets
            train_y_pred_prob = model.predict_proba(self.train_x)[:,1]
            test_y_pred_prob = model.predict_proba(self.test_x)[:,1]
            model_name = model.__class__.__name__

            train_roc_auc = np.round(roc_auc_score(self.train_y, train_y_pred_prob),4) #Calculate Roc_auc score
            test_roc_auc = np.round(roc_auc_score(self.test_y, test_y_pred_prob),4) #Calculate Roc_auc score
            
            train_fpr,train_tpr,train_threshold=roc_curve(self.train_y,train_y_pred_prob)
            precision_train,recall_train,_ =  precision_recall_curve(self.train_y, train_y_pred_prob) 
            precision_test,recall_test,_ = precision_recall_curve(self.test_y, test_y_pred_prob)
            
            pr_auc_train_score = np.round(auc(recall_train, precision_train),4)
            pr_auc_test_score =  np.round(auc(recall_test, precision_test),4)

            best_thres = self.find_best_threshold(threshold= train_threshold,fpr = train_fpr, tpr = train_tpr)
            train_pred = self.predict_with_best_t(train_y_pred_prob, best_thres)
            test_pred = self.predict_with_best_t(test_y_pred_prob, best_thres)
            
            # for training
            train_f1_macro = np.round(f1_score(self.train_y, train_pred, average = 'macro'),4) # calculating macro f1 score
            train_f1_each_class_scores = np.round(f1_score(self.train_y, train_pred, average = None  ),4) # calculating f1 score for each class
            train_precision = np.round(precision_score(self.train_y, train_pred),4) # Calculate Precision
            train_recall = np.round(recall_score(self.train_y, train_pred),4)  # Calculate Recall

            #for testing
            test_f1_macro = np.round(f1_score(self.test_y, test_pred, average = 'macro'),4) #calculate the macro_f1 score
            test_f1_each_class_scores = np.round(f1_score(self.test_y, test_pred, average = None),4) # calculating f1 score for each class
            test_precision = np.round(precision_score(self.test_y, test_pred),4) # Calculate Precision
            test_recall = np.round(recall_score(self.test_y, test_pred),4)  # Calculate Recall
            logging.info(f"\n\nClassificatin metrics for the model '{model.__class__.__name__}': ")
            table = PrettyTable(["set","macro_f1_score","each_class_f1_score","precision_score","recall_score","roc_auc_score","pr_auc_score"])
            table.add_row(["train",train_f1_macro,train_f1_each_class_scores, train_precision,train_recall,train_roc_auc,pr_auc_train_score])
            table.add_row(["test",test_f1_macro,test_f1_each_class_scores, train_precision,test_recall,test_roc_auc,pr_auc_test_score])
                        
            logging.info(f"{table}")


            model_pr_auc = pr_auc_test_score
            diff_test_train_roc_auc= abs(train_roc_auc - test_roc_auc)

            logging.info(f"Diff of  test, train roc_auc: [{diff_test_train_roc_auc}].") 
            metric_info_artifact = MetricInfoArtifact(model_name= model_name,
                                                    model_object=model,
                                                    train_recall= train_recall,
                                                    test_recall=test_recall,
                                                    train_f1_each_class_scores=train_f1_each_class_scores,
                                                    test_f1_each_class_scores=test_f1_each_class_scores,
                                                    train_precision= train_precision,
                                                    test_precision= test_precision,
                                                    train_macro_F1= train_f1_macro,
                                                    test_macro_F1= test_f1_macro,
                                                    train_roc_auc=train_roc_auc,
                                                    test_roc_auc=test_roc_auc,
                                                    pr_auc_train=pr_auc_train_score,
                                                    pr_auc_test=pr_auc_test_score)
            return metric_info_artifact
    
        except Exception as e:
            logging.info(BaseException((e,sys)))
            raise BackorderException(e,sys)