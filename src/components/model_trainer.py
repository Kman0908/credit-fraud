import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, evaluate_model

@dataclass 
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_training(self, train_arr, test_arr):
        try:
            logging.info('Training initiated')

            logging.info('Splitting data')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            sm = SMOTE()

            logging.info('Resampling data using SMOTE(Synthetic Minority Over-sampling Technique)')
            X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
            models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForestClassifier': RandomForestClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'XGBClassifier': XGBClassifier(),
                'CatBooostClassifier': CatBoostClassifier(verbose = False)
            }

            report: dict = evaluate_model(X_resampled, X_test, y_resampled, y_test, models)

            best_score = max(sorted(report.values()))
            best_model = list(report.keys())[
                list(report.values()).index(best_score)
            ]

            print(f'Best model: {best_model}, w/score: {best_score}')
            logging.info(f'Best model: {best_model}, w/score: {best_score}')

            print(f'Model report:\n {report}')
            logging.info(f'Model report:\n{report}')

            save_obj(self.model_trainer_config.model_path, models[best_model])  
            logging.info('Model saved')

        except Exception as e:
            logging.exception('Error occurred at Model Trainer')
            raise CustomException(e, sys)