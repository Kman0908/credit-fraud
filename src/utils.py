import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import f1_score
import pickle

def save_obj(path, obj):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.exception('Error occurred at utils.save_obj')
        raise CustomException(e, sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models: dict):
    report = {}
    try:
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            # Receiver Operating Characteristic - Area Under the Curve
            score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = score

        return report
    except Exception as e:
        logging.exception('Error occurred at utils.evaluate_model')
        raise CustomException(e, sys)

def load_obj(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.exception('Error occurred at utils.load_obj')
        raise CustomException(e, sys)