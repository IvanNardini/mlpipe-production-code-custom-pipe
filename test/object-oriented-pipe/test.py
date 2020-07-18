#Read Data
import pandas as pd

#Pipeline
from pipeline import *

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

pipeline = Pipeline(
                    dropped_columns=config['dropped_columns'],
                    renamed_columns=config['renamed_columns'],
                    target=config['target'],
                    missing_predictors=config['missing_predictors'],
                    nominal_predictors=config['nominal_predictors'],
                    features=config['features'], 
                    features_selected=config['features_selected'],
                    binning_meta=config['binning_meta'],
                    encoding_meta=config['encoding_meta'],
                    dummies_meta=config['dummies_meta'],
                    test_size=0.1
                    )

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Read configuration
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)
    
    logging.info('Training process started!')
    df = pd.read_csv(config['paths']['data_path'])
    pipeline.fit(df)
    logging.info('Training process successfully completed!')
    
    print("*"*20)
    print("Model Assessment".center(20, '*'))
    print("*"*20)
    pipeline.evaluate()
    print("*"*20)
    print("Model Predictions".center(20, '*'))
    print("*"*20)

    logging.info('Scoring process started!')
    predictions = pipeline.predict(df)
    logging.info('Scoring process successfully completed!')

    print('First 20 prediticions are: {}'.format(predictions[:10]))