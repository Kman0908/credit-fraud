import os
import sys

from src.exception import CustomException
from src.logger import logging

import pickle

def save_obj(path, obj):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.exception('Error occurred at utils.save_obj')
        raise CustomException(e, sys)
