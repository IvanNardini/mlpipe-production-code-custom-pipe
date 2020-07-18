'''
pipeline modules contains the pipeline object
'''
from templates.data_preprocessing import Preprocessing
from templates.modelling import Model

# Data Preparation
import pandas as pd
import numpy as np

#Feature enginnering
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Pipeline(Preprocessing, Model):   
    
    def __init__(self, dropped_columns, renamed_columns, target, nominal_predictors, features, features_selected, binning_meta, encoding_meta, dummies_meta, test_size):

        ##Data
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        ##Declared Variables
        self.dropped_columns = dropped_columns
        self.renamed_columns = renamed_columns
        self.target = target
        self.nominal_predictors = nominal_predictors
        self.features = features
        self.features_selected = features_selected
        self.binning_meta = binning_meta
        self.encoding_meta = encoding_meta
        self.dummies_meta = dummies_meta

        ##Engineering metadata
        self.missing_predictors = []
        self.random_state = 0
        self.test_size = test_size

    # =====================================================================================================

    #fit pipeline
    def fit(self, data):

        #Initialize
        self.data = data
        self.missing_predictors = [col for col in self.data.select_dtypes(include='object').columns if any(self.data[col].str.contains('?', regex=False))]
        #Step1: Arrange Data
        self.data = self.Data_Preparer(self.data, self.dropped_columns, self.renamed_columns)
        #Step2: Impute missing
        self.data = self.Missing_Imputer(self.data, self.missing_predictors, replace='missing')
        #Step3: Binning Variables
        self.data = self.Binner(self.data, self.binning_meta)
        #Step4: Encoding Variables
        self.data = self.Encoder(self.data, self.encoding_meta)
        #Step5: Generate Dummies
        self.data = self.Dumminizer(self.data, self.nominal_predictors, self.dummies_meta)
        #Step6: Scale Features
        self.data = self.Scaler(self.data, self.features)
        #Step7: Balancing
        self.X, self.y = self.Balancer(self.data, self.features_selected, self.target, self.random_state)
        #Step8: Split for training
        self.X_train, self.X_test, self.y_train, self.y_test = self.Data_Splitter(self.X, self.y,
                                                                                  test_size = self.test_size,
                                                                                  random_state = self.random_state)
        #Step9: Model Fit 
        self.Model.fit(self.X_train, self.y_train)

        return self


    def transform(self, data):
        pass

    def predict(self, data):
        pass

    def evaluate(self, data):
        pass
