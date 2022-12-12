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
            logging.info(f"after numerical_transformation: {X}")
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

    def predict(x):
        try:
            X = self.transformer.transform(x)
            prediction = self.model.predict(X)
            return prediction
        except Exception as e:
            logging.info(BackoderModel(e,sys))
            BackorderException(e,sys)
    
    def __repr__(self):
        return f"{type(self.model).__name__}()"

    def __str__(self):
        return f"{type(self.model).__name__}()"