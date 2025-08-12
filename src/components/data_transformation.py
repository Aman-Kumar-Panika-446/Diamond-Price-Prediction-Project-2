import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_obj

# Data Transformation Config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')



# Data Transformation Class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation Initiated')
            
            # Segregating the numeric and categoric columns
            numeric_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_col = ['cut', 'color', 'clarity']

            # Defining Order for these Ordinal features 
            cut_map = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            clarity_map = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            color_map = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        
            logging.info("Pipeline Initiated")

            # Numerical Pipelin
            num_pipeline = Pipeline(

                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[
                        cut_map, color_map, clarity_map,])),
                    ('scaler', StandardScaler())
                ]
            )

            # Column Transformer to combine both pipelines
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_col),
                    ('cat_pipeline', cat_pipeline, categorical_col)
                ]
            )
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error Occured in Data Transformation")
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading of Train & Test dataset has been completed")
            logging.info('Obtaining preprocessor object... ')

            preprocessor_obj = self.get_data_transformation_object()
            
            logging.info('Obtained preprocessor object ')
            target_col = 'price'
            drop_col = [target_col,'id']

            logging.info("Applying preprocessing object on dataset")
            X_train = train_df.drop(drop_col, axis=1)
            Y_train = train_df[target_col]

            X_test = test_df.drop(drop_col, axis=1)
            Y_test = test_df[target_col]

            X_train_processed = preprocessor_obj.fit_transform(X_train)
            X_test_processed = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_processed, np.array(Y_train)]
            test_arr = np.c_[X_test_processed, np.array(Y_test)]

            logging.info("Preprocessing Successfully Applied")

            save_obj(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Pickle file is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured initialization of Data Transformation ")
            raise CustomException(e,sys)
