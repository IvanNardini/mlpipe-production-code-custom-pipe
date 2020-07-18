# Data Preparation
import numpy as np
import pandas as pd

#Model
from sklearn.ensemble import RandomForestClassifier

class Model():

    def Model(self):
        '''
        Train the model and store it
        :params: X_train, y_train, output_path
        :return: None
        '''
        # initialise the model
        rfor = RandomForestClassifier(max_depth=25, 
                                    min_samples_split=5, 
                                    n_estimators=300,
                                    random_state=9)
        return rfor