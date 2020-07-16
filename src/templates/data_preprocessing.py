# Data Preparation
import numpy as np
import pandas as pd

# Model Training
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

class Preprocessing():

    def data_preparer(self, data, dropped_columns, renamed_columns):
        '''
        Drop and Rename columns
        :params: data, columns_to_drop
        :return: DataFrame
        '''
        data = data.copy()
        data.drop(dropped_columns, axis=1, inplace=True)
        data.rename(columns=renamed_columns, inplace=True)
        return data