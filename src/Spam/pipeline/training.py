from Spam.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from Spam.entity.artifact_entity import DataIngestionArtifact
from Spam.exception import SpamException
import sys,os
from Spam.logger import logging
from Spam.components.data_ingestion import DataIngestion

class TrainPipeline:
    def __init__(self):

        self.training_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
    
    def start_data_ingestion(self)->DataIngestionArtifact:

        try:
            
            logging.info("Starting Data Ingestion")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion completed. The artifact location:- {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise SpamException(e,sys)
    
    def run_pipeline(self):
        try:
            logging.info(f"Starting re-training pipeline")
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()

        except Exception as e:
            raise SpamException(e,sys)  