import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj = DataTransformationConfig()
    
    def get_preprocessor_obj(self):
        try:
            cols = ['Time', 'Amount']
            preprocessor = ColumnTransformer(transformers = [
                ('Scalar', StandardScaler(), cols)],
                remainder = 'passthrough'
            )

            return preprocessor
        except Exception as e:
            logging.exception('Exception occurred at Data Transformation')
            raise CustomException(e, sys)
    
    def initiate_preprocessing(self, train_path, test_path):
        try:
            test = pd.read_csv(test_path)
            train = pd.read_csv(train_path)

            logging.info('Data loaded as pandas dataset')
            logging.info(f'Test:\n{test.head()}')
            logging.info(f'Train:\n{train.head()}')

            preprocessor = self.get_preprocessor_obj()
            logging.info('Got preprocessor object')

            target_col = 'Class'

            input_train  = train.drop(columns = target_col)
            target_train = train[target_col]

            input_test = test.drop(columns = target_col)
            target_test = test[target_col]

            logging.info('Applying preprocessor')
            train_transformed = preprocessor.fit_transform(input_train)
            test_transformed = preprocessor.transform(input_test)

            train_arr = np.c_[train_transformed, np.array(target_train)]
            test_arr = np.c_[test_transformed, np.array(target_test)]

            logging.info('Saved preprocessor object')
            save_obj(self.preprocessor_obj.preprocessor_path, preprocessor)
            return(
                train_arr,
                test_arr,
                self.preprocessor_obj.preprocessor_path
            )
        except Exception as e:
            logging.exception('Error occurred at Data transformation')
            raise CustomException(e, sys)