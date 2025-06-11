from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from notebooks.transform import data_corr_coef

import sys, os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, available_columns):
        try:
            logging.info('Data Transformation initiated')

            predefined_numerical_cols = [
                'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone',
                'Benzene', 'Toluene', 'RH', 'WS', 'WD', 'SR', 'BP', 'AT', 'RF', 'TOT-RF'
            ]

            numerical_cols = [col for col in predefined_numerical_cols if col in available_columns]
            missing_cols = [col for col in predefined_numerical_cols if col not in available_columns]
            if missing_cols:
                logging.warning(f"The following expected numerical columns are missing: {missing_cols}")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols)
            ])

            logging.info('Data Transformation pipeline created successfully.')
            return preprocessor

        except Exception as e:
            logging.error('Error occurred during data transformation object creation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            target_column = 'AQI'

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Train and test data read successfully.')

            train_df = train_df.dropna(subset=[target_column])
            test_df = test_df.dropna(subset=[target_column])

            drop_columns = data_corr_coef()
            if target_column not in drop_columns:
                drop_columns.append(target_column)

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            preprocessing_obj = self.get_data_transformation_object(input_feature_train_df.columns)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessing completed and preprocessor object saved.')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error('Error during initiate_data_transformation')
            raise CustomException(e, sys)
