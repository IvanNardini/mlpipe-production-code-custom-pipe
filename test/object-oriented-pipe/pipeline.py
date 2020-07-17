'''
pipeline modules contains the pipeline object
'''
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
    
    def __init__(self, dropped_columns, renamed_columns, missing_predictors, binning_meta, encoding_meta):

        #Data
        self.data = None

        #Global Variables (declared)
        self.dropped_columns = dropped_columns
        self.renamed_columns = renamed_columns
        self.binning_meta = binning_meta
        self.encoding_meta = encoding_meta

        # self.target = target
        # self.predictors = predictors
        
        self.numerical_predictors = []
        self.discrete_predictors = []
        self.continuous_predictors = []
        self.categorical_predictors = []
        self.ordinal_predictors = []
        self.nominal_predictors = []
        self.binned_variables = []
        self.encode_variables = []
        self.features = []
        self.features_selected = []

        #Engineering metadata (derived)
        self.missing_predictors = []
        self.dummies_meta = {}

    #fit pipeline
    def fit(self, data):

        #MetaData
        self.data = data
        self.missing_predictors = [col for col in self.data.select_dtypes(include='object').columns if any(self.data[col].str.contains('?', regex=False))]

        # =====================================================================================================

        #Step1: Arrange Data
        self.data = Preprocessing.data_preparer(self, self.data, self.dropped_columns, self.renamed_columns)
        #Step2: Impute missing
        self.data = Preprocessing.missing_imputer(self, self.data, self.missing_predictors, replace='missing')
        #Step3: Binning Variables
        self.data = Preprocessing.binner(self, self.data, self.binning_meta)
        # #Step4: Encoding Variables
        self.data = Preprocessing.encoder(self, self.data, self.encoding_meta)
        return self


    def transform(self, data):
        pass

    def predict(self, data):
        pass