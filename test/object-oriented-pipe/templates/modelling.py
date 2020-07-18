# Data Science 
import numpy as np
import pandas as pd

#Model
from sklearn.ensemble import RandomForestClassifier

class Model():

    def Model_RFor(self, max_depth, min_samples_split, n_estimators):
        '''
        Train the model and store it
        :params: X_train, y_train, output_path
        :return: None
        '''
        # initialise the model
        rfor = RandomForestClassifier(max_depth=max_depth, 
                                    min_samples_split=min_samples_split, 
                                    n_estimators=n_estimators,
                                    random_state=9)
        return rfor