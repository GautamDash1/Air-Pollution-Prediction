import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def categorize_aqi(self, aqi_values):
        bins = [0, 50, 100, 200, 300, 400, float('inf')]
        labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        return pd.cut(aqi_values, bins=bins, labels=labels)

    def evaluate_model(self, X_train, y_train, X_test, y_test, models: dict):
        try:
            report = {}
            for name, model in models.items():
                try:
                    logging.info(f"Training model: {name}")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mse ** 0.5

                    report[name] = {
                        "R2": r2,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse
                    }

                except Exception as e:
                    logging.error(f"Model {name} training failed: {str(e)}")
            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor(),
                'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100)
            }

            model_report: dict = self.evaluate_model(X_train, y_train, X_test, y_test, models)

            print("\nModel Performance Metrics:\n")
            for model_name, metrics in model_report.items():
                print(f"{model_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                print("---------------------------------------------------")

            logging.info(f'Model Report : {model_report}')

            best_model_name = max(model_report, key=lambda x: model_report[x]['R2'])
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]['R2']

            print(f"\nBest Model Found: {best_model_name} with R2 Score: {best_model_score:.4f}")
            logging.info(f'Best Model Found: {best_model_name} with R2 Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
