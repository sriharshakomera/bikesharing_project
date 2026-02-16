import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info('Data Transformation initiated')
            numerical_columns=['season',  'mnth', 'hr', 'holiday', 'weekday', 'workingday','weathersit', 
            'temp', 'atemp', 'hum', 'windspeed'] #'yr'
            categorical_columns=[]

            logging.info('Pipeline Initiated')
            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("numerical columns scaling completed !!")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('ordinalencoder',OrdinalEncoder(categories=[])),
                    ("scaler",StandardScaler()),
                ]

            )
            logging.info("Categorical columns encoding completed !!")
            
            logging.info(f"Categorical columns:{categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data is complete")

            logging.info("obtaining preprocessor object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='cnt'
            drop_columns = [target_column_name,'dteday','yr']
           # numerical_columns=['temp','atemp','hum','windspeed']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            logging.info( print("train_df columns: ",train_df.columns))
            logging.info( print("test_df columns: ",test_df.columns))

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info( print("input_feature_test_df columns: ",input_feature_test_df.columns))

            logging.info(f"Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )       


        except Exception as e:
            raise CustomException(e,sys)

