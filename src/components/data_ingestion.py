import os
import sys
from src.exceptions import CustomExceptions
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransfromation

@dataclass
class DataIngestionConfig:
    train_path:str=os.path.join("artifact","train.csv")
    test_path:str=os.path.join("artifact","test.csv")
    raw_path:str=os.path.join('artifact',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def Initiate_data_ingestion(self):
        logging.info("Staring the Ingestion process")
        try:
            df=pd.read_csv("/Users/vijayrakeshreddybandela/Documents/machine_learning(vs)/mlprojects/src/notebook/data/stud.csv")
            logging.info("Read the data using pandas")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_path,index=False,header=True)
            logging.info("Loaded into raw dataset")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=30)
            logging.info("Split into train and test set")
            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            logging.info("Loaded into Train dataset")
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)
            logging.info("Loaded into Test dataset")

            return (self.ingestion_config.train_path,self.ingestion_config.test_path,self.ingestion_config.raw_path)
        except Exception as e:
            raise CustomExceptions(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_path, test_path, raw_path = obj.Initiate_data_ingestion()

    data_trans = DataTransfromation()
    data_trans.initiate_ttransformation(train_path, test_path)

