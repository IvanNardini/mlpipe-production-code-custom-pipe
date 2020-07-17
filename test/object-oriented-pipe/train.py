#Pipeline
from pipeline import *

#Utils
import logging
import ruamel.yaml as yaml
import warnings
warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

# Read configuration
stream = open('config.yaml', 'r')
config = yaml.load(stream)

pipeline = Pipeline(
                    dropped_columns=config['dropped_columns'],
                    renamed_columns=config['renamed_columns'],
                    missing_predictors=config['missing_predictors'], 
                    binning_meta=config['binning_meta']
                )

if __name__ == "__main__":

    import logging
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Training process started!')

    df = pd.read_csv(config['paths']['data_path'])
    test = pipeline.fit(df)
    print(test.data.head(20))