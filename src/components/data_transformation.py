import os
import sys
from src.exceptions import CustomExceptions
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preposseror_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransfromation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:    
            numerical_features=["writing_score","reading_score"]
            categorical_features=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("creating numerical pipeline")
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("creating categorical pipeline")

            preprocessor=ColumnTransformer(
                [
                    ("numerical pipeline",num_pipeline,numerical_features),
                    ("categorical pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomExceptions(e,sys)
        
    def initiate_ttransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_obj()
            target_column = "math_score"
            numerical_features = ["writing_score", "reading_score"]
            
            input_features_train = train_df.drop(columns=target_column, axis=1)
            target_features_train = train_df[target_column]

            input_features_test = test_df.drop(columns=target_column, axis=1)
            target_features_test = test_df[target_column]

            print("Columns in input_features_train:", input_features_train.columns)  # Debugging print

            logging.info("Applying preprocessing")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_features_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_features_test)]

            logging.info("Saved processing object")
            save_obj(
                file_path=self.data_transformation_config.preposseror_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preposseror_obj_file_path
            )
            
        except Exception as e:
            raise CustomExceptions(e, sys)


