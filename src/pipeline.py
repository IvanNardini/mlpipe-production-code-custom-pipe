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
        preparer = Preprocessing.data_preparer(data, self.dropped_columns, self.renamed_columns)
        return preparer

    #Step2: Impute missings
    def Impute_Missing(self, data):
        imputer = Preprocessing.missing_imputer(data, )


    # # #Step2: Impute Missings
    # # def missing_imputer(self, df, var, replace='missing'):
    # #     '''
    # #     Imputes '?' character with 'missing' label
    # #     :params: data, var, replace
    # #     :return: Series
    # #     '''
    # #     df = df.copy()

    # #     return data[var].replace('?', replace)

    