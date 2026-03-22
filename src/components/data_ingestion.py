import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion started')
        try:
            logging.info('Read data as pandas dataframe')
            data = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'creditcard.csv'))
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok = True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)

            train_data, test_data = train_test_split(data, random_state = 42, test_size = 0.2)

            train_data.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)

            logging.info('Data ingestion complete')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.exception('Exception occurred at data ingestion')
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()