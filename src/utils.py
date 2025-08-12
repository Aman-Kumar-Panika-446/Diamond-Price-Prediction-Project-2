import os, sys
import pickle
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException

def save_obj(file_path, obj):
    
    try: 
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Error while creation of pickle file")
        raise CustomException(e,sys)


def evaluate_model(X_train, X_test, Y_train, Y_test, models):
    
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            score = r2_score(Y_test, Y_pred)

            report[list(models.keys())[i]] = score
        return report
    except Exception as e:
        logging.info("Exception Occured while computing the accuracy of model")
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
