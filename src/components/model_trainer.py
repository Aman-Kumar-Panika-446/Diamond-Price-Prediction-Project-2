import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.utils import save_obj
from src.utils import evaluate_model

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting the Independent & Dependent variables")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': Ridge(),
                    'Lasso Regression': Lasso(),
                    'ElasticNet Regression': ElasticNet()
                    } 

            model_report:dict = evaluate_model(X_train, X_test, Y_train, Y_test, models)
            print(model_report)
            

            print('='*50)

            logging.info(f"Model Report: {model_report}")

            # to get best model score
            max_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(max_score)
            ]

            best_model = models[best_model_name]
            print(f"The Best Model is: {best_model_name}, R2 Score = {max_score}")
            logging.info(f"The Best Model is: {best_model_name}, R2 Score = {max_score}")

            save_obj(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured while training of model")
            raise CustomException(e,sys)