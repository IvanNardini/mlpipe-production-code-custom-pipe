# Data Preparation
import pandas as pd
import numpy as np

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Model Deployment
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Pipeline:   
    
    def __init__(self, dropped_columns, renamed_columns):

        #Global Variables
        self.dropped_columns = dropped_columns
        self.renamed_columns = renamed_columns
        # self.target = target
        # self.predictors = predictors
        
        #Engineering metadata (coming from the data)
        # self.missing_predictors = {}
        # self.binning_meta = {}
        # self.dummies_meta = {}
        # self.encoding_meta = {}

    
    #Step1: Arrange Data
    def data_preparer(self, data):
        '''
        Drop and Rename columns
        :params: data, columns_to_drop
        :return: DataFrame
        '''
        data = data.copy()
        data.drop(self.dropped_columns, axis=1, inplace=True)
        data.rename(columns=self.renamed_columns, inplace=True)
        return data


    # #Step2: Impute Missings
    # def missing_imputer(self, df, var, replace='missing'):
    #     '''
    #     Imputes '?' character with 'missing' label
    #     :params: data, var, replace
    #     :return: Series
    #     '''
    #     df = df.copy()

    #     return data[var].replace('?', replace)

    