from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler
from backorder.exception import BackorderException
from sklearn.impute import SimpleImputer
import os, sys
from backorder.logger import logging
import pandas as pd

class TargetValueMapping:
    def __init__(self):
        self.Yes: int = 1
        self.No: int = 0

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class FeatureScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self,X,y=None):
        try:
            X =np.log(X+0.0001)
            self.scaler.fit(X,y)
            return self
        except Exception as e:
            BackorderException(e,sys)
    
    def log_transform(self,data_frame:pd.DataFrame):
        """
        np.sign(x) => Returns an element-wise indication of the sign of a number.
                      The sign function returns -1 if x < 0,
                       0 if x==0,
                       1 if x > 0.
                       nan is returned for nan inputs.
        Applies log transform on input data"""
        sign = np.sign(data_frame)
        data_frame =  np.log(1.0+abs(data_frame))*sign
        return data_frame

    def fit_transform(self, X, y=None):
        try:
            X = np.apply_along_axis(self.log_transform, 1, X)
            X = self.scaler.fit_transform(X,y)
            return X
        except Exception as e:
            BackorderException(e,sys)
        
    def transform(self,X):
        try:
            X = self.scaler.transform(X)
            return X
        except Exception as e:
            BackorderException(e,sys)


class NullImputer(BaseEstimator, TransformerMixin):
    def __init__(self,num_features_list):
        try:
            self.num_features_list = num_features_list
            self.imputer = SimpleImputer(strategy="median")
        except Exception as e:
            BackorderException(e,sys) 

    def fit(self,X,y=None):
        try:
            X1 =X[self.num_features_list]
            self.imputer.fit(X=X1)
            return self
        except Exception as e:
            BackorderException(e,sys) 

    def transform(self,X,y=None):
        try:
            X1 =X[self.num_features_list]
            X[self.num_features_list] = self.imputer.transform(X=X1)
            return X
        except Exception as e:
            BackorderException(e,sys) 

    def fit_transform(self,X,y=None):
        try:
            X1 =X[self.num_features_list]
            X[self.num_features_list] = self.imputer.fit_transform(X=X1)
            return X
        except Exception as e:
            BackorderException(e,sys) 

class DataPreprocessor:
    def __init__(self, imputer, preprocessor,num_features_list):
        try:
            self.num_features_list = num_features_list
            self.imputer = imputer
            self.preprocessor = preprocessor
        except Exception as e:
            BackorderException(e,sys)
    
    def transform(self,X,y=None):
        try:
            X1 =X[self.num_features_list]
            X[self.num_features_list] = self.imputer.transform(X=X1)
            X = self.preprocessor.transform(X)
            return X
        except Exception as e:
            BackorderException(e,sys)

class BackoderModel:
    def __init__(self, transformer_obj, model_obj):
        self.transformer = transformer_obj
        self.model = model_obj

    def transform_predict(self,x):
        try:
            X = self.transformer.transform(x)
            prediction = self.model.predict(X)
            return prediction
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)
    
    def predict(self,x):
        try:
            prediction = self.model.predict(x)
            return prediction
        except Exception as e:
            logging.info(BackorderException(e,sys))
            BackorderException(e,sys)

    def predict_proba(self,x):
        prediction = self.model.predict_proba(x)
        return prediction
    def __repr__(self):
        return f"{type(self.model).__name__}()"

    def __str__(self):
        return f"{type(self.model).__name__}()"

class BackorderData:

    def __init__(self,
                national_inv:float,
                lead_time:float,
                in_transit_qty:float,
                forecast_6_month:float,
                sales_6_month:float,
                min_bank:float,
                potential_issue:str,
                pieces_past_due:float,
                perf_6_month_avg:float,
                local_bo_qty:float,
                deck_risk:str,
                oe_constraint:str,
                ppap_risk:str,
                stop_auto_buy:str,
                rev_stop:str,
                 ):

        try:
                self.national_inv= national_inv
                self.lead_time=lead_time
                self.in_transit_qty=in_transit_qty
                self.forecast_6_month=forecast_6_month
                self.sales_6_month=sales_6_month
                self.min_bank=min_bank
                self.potential_issue=potential_issue
                self.pieces_past_due=pieces_past_due
                self.perf_6_month_avg=perf_6_month_avg
                self.local_bo_qty=local_bo_qty
                self.deck_risk=deck_risk
                self.oe_constraint=oe_constraint
                self.ppap_risk=ppap_risk
                self.stop_auto_buy=stop_auto_buy
                self.rev_stop=rev_stop
            
        except Exception as e:
            raise BackorderException(e, sys) from e


    def get_backorder_input_data_frame(self):
        try:
            backorder_input_dict = self.get_backorder_data_as_dict()
            return pd.DataFrame(backorder_input_dict)

        except Exception as e:
            raise BackorderException(e, sys) from e


    def get_backorder_data_as_dict(self):
        try:
            input_data = {
               "national_inv": [self.national_inv],
               "lead_time" :[self.lead_time],
                "in_transit_qty" :[self.in_transit_qty],
                "forecast_6_month": [self.forecast_6_month],
                "sales_6_month":[self.sales_6_month],
                "min_bank":[self.min_bank],
                "potential_issue":[self.potential_issue],
                "pieces_past_due":[self.pieces_past_due],
                "perf_6_month_avg":[self.perf_6_month_avg],
                "local_bo_qty":[self.local_bo_qty],
                "deck_risk":[self.deck_risk],
                "oe_constraint":[self.oe_constraint],
                "ppap_risk":[self.ppap_risk],
                "stop_auto_buy":[self.stop_auto_buy],
                "rev_stop":[self.rev_stop],
                }

            return input_data

        except Exception as e:
            raise BackorderException(e, sys) 