'''
pipeline modules contains the pipeline object
'''
from templates.data_preprocessing import Preprocessing
from templates.features_engineering import FeatureEngineering
from templates.modelling import Models
from templates.postprocessing import PostProcessing

#Utils
import logging
import joblib
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


class Pipeline():   
    
    def __init__(self, dropped_columns, renamed_columns, missing_predictors, target, predictors, target_encoding):

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
        self.anomalies = 'umbrella_limit'
        self.missing_predictors = missing_predictors
        self.target = target
        self.predictors = predictors
        self.target_encoding = target_encoding
        # self.encoding_meta = encoding_meta
        # self.features = features
        # self.features_selected = features_selected
        # self.binning_meta = binning_meta
        
        # self.dummies_meta = dummies_meta

        ## metadata
        self.replace = 'missing'
        self.test_size = 0.1
        self.random_state_sample = 1
        # self.random_state_smote = 9
        # self.random_state_model = 8
        # self.test_size = test_size
        # self.max_depth = 25
        # self.min_samples_split = 5
        # self.n_estimators= 300

        # #Model
        # self.model = None

    # =====================================================================================================

    #fit pipeline
    def fit(self, data):

        #Initialize
        self.data = data
        #Step1: Drop columns 
        self.data = Preprocessing.dropper(self, self.data, self.dropped_columns)
        #Step2: Rename columns 
        self.data = Preprocessing.renamer(self, self.data, self.renamed_columns)
        #Step3: Remove anomalies
        self.data = Preprocessing.anomalizier(self, self.data, self.anomalies)
        #Step4: Impute missing
        self.data = Preprocessing.missing_imputer(self, self.data, 
                                                  self.missing_predictors,
                                                  replace=self.replace)
        #Step6: Split data
        self.X_train, self.X_test, self.y_train, self.y_test = Preprocessing.data_splitter(
                                                    self,
                                                    self.data,
                                                    self.target,
                                                    self.predictors,
                                                    self.test_size,
                                                    self.random_state_sample
                                                    )
        #Step7: Encode Target
        self.y_train = FeatureEngineering.target_encoder(self, self.y_train, self.encoding_meta)

        
    #     self.data = Preprocessing.Binner(self, self.data, self.binning_meta)
    #     #Step4: Encoding Variables
    #     self.data = Preprocessing.Encoder(self, self.data, self.encoding_meta)
    #     #Step5: Generate Dummies
    #     self.data = Preprocessing.Dumminizer(self, self.data, self.nominal_predictors, self.dummies_meta)
    #     #Step6: Scale Features
    #     self.data = Preprocessing.Scaler(self, self.data, self.features)
    #     #Step7: Balancing
    #     self.X, self.y = Preprocessing.Balancer(self, self.data, self.features_selected, self.target, self.random_state_smote)
    #     #Step8: Split for training
    #     self.X_train, self.X_test, self.y_train, self.y_test = Preprocessing.Data_Splitter(self, self.X, self.y,
    #                                                                               test_size = self.test_size,
    #                                                                               random_state = self.random_state_sample)
    #     #Step9: Model Fit 
    #     self.model = Models.RFor(self, max_depth=self.max_depth, 
    #                     min_samples_split=self.min_samples_split, 
    #                     n_estimators=self.n_estimators, random_state=self.random_state_model)
    #     self.model.fit(self.X_train, self.y_train)

    #     return self

    # #transform data
    # def transform(self, data):
    #     data = data.copy()
    #     #Step1: Arrange Data
    #     data = Preprocessing.Data_Preparer(self, data, self.dropped_columns, self.renamed_columns)
    #     #Step2: Impute missing
    #     data = Preprocessing.Missing_Imputer(self, data, self.missing_predictors, replace='missing')
    #     #Step3: Binning Variables
    #     data = Preprocessing.Binner(self, data, self.binning_meta)
    #     #Step4: Encoding Variables
    #     data = Preprocessing.Encoder(self, data, self.encoding_meta)
    #     #Step5: Generate Dummies
    #     data = Preprocessing.Dumminizer(self, data, self.nominal_predictors, self.dummies_meta)
    #     #Step6: Scale Features
    #     data = Preprocessing.Scaler(self, data, self.features)
    #     #Step7: Select Features
    #     data = data[self.features_selected]
    #     return data

    # #predict
    # def predict(self, data):
    #     #Step1: Engineer the data
    #     data = self.transform(data)
    #     #Step2: Predict
    #     predictions = self.model.predict(data)
    #     return predictions
        
    # #evaluate
    # def evaluate(self):
    #     PostProcessing.evaluate_classification(self, self.model, self.X_train, self.y_train, 
    #                                  self.X_test, self.y_test)