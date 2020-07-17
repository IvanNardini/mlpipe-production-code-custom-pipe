from templates.data_preprocessing import Preprocessing

# Data Preparation
import pandas as pd
import numpy as np

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Pipeline(Preprocessing):   
    
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
    def Prepare_Variables(self, data):
        preparer = Preprocessing.data_preparer(self, data, self.dropped_columns, self.renamed_columns)
        return preparer

    # def data_preparer(self, data):
    #     '''
    #     Drop and Rename columns
    #     :params: data, columns_to_drop
    #     :return: DataFrame
    #     '''
    #     data = data.copy()
    #     data.drop(self.dropped_columns, axis=1, inplace=True)
    #     data.rename(columns=self.renamed_columns, inplace=True)
    #     return data


    # # #Step2: Impute Missings
    # # def missing_imputer(self, df, var, replace='missing'):
    # #     '''
    # #     Imputes '?' character with 'missing' label
    # #     :params: data, var, replace
    # #     :return: Series
    # #     '''
    # #     df = df.copy()

    # #     return data[var].replace('?', replace)

    