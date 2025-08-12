import sys, os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.pipelines.training_pipeline import TrainingPipeline


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, feature):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                training_pipeline = TrainingPipeline()
                training_pipeline.run_pipeline()

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            scaled_data = preprocessor.transform(feature)
            prediction = model.predict(scaled_data)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, carat: float, depth:float, table: float, x:float, y:float, z:float, cut:str, color:str, clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)

