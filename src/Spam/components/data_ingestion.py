from Spam.exception import SpamException
from Spam.logger import logging
from Spam.entity.config_entity import DataIngestionConfig
from Spam.entity.artifact_entity import DataIngestionArtifact
from pathlib import Path

from pandas import DataFrame
from Spam.data_access.get_data import NewsData
from Spam.utils.common import read_yaml
from Spam.constants.training_pipeline import SCHEMA_FILE_PATH

from sklearn.model_selection import train_test_split
import os,sys


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            super().__init__()
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml(Path(SCHEMA_FILE_PATH))
        except Exception as e:
            raise SpamException(e,sys)

    def export_data_into_feature_store(self) -> DataFrame:
        
        try:
            logging.info("Exporting data from MongoDB")
            logging.info(self.data_ingestion_config.collection_name)
            news_data = NewsData()
            dataframe = news_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info("Exported data from MongoDB")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path            

            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info("Saving data into feature store directory")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except  Exception as e:
            raise  SpamException(e,sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """

        try:
            # logging.info("Performing train test split on the dataframe")
            
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            # logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            # logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise SpamException(e,sys)
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:

            dataframe = self.export_data_into_feature_store()
            logging.info(dataframe.shape)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact

        except Exception as e:
            raise SpamException(e,sys)