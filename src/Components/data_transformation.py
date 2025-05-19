from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import sys,os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    def __init__(self):
        preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            numerical_cols=['co','no','no2','so2','pm2_5','pm10']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols)
            ])

            logging.info('Data Transformation completed')

        except Exception as e:
            logging.info('Error occured at data transformation')
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Read of train and test dataset completed')


            logging.info('Obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformation_object()

            drop_column=['o3','nh3']

            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr]
            test_arr=np.c_[input_feature_test_arr]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            logging.info('Applying preprocessing object on training and testing data')
            
        except Exception as e:
            raise CustomException(e,sys)