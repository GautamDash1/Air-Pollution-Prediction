import os
import sys
import pandas as pd
from src.Components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation=DataTransformation()
    train_arr,test_arr,obj_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
