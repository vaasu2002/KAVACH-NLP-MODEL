import sys
from typing import Optional
import numpy as np
import pandas as pd
from Spam.config.mongo_db_connection import MongoDBClient
from Spam.constants.database import DATABASE_NAME,SPAM_COLLECTION_NAME
from Spam.exception import SpamException

class NewsData:

    def __init__(self):

        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        
        except Exception as e:
            raise SpamException(e, sys)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        
        try:

            if database_name is None:
                collection = self.mongo_client.database[collection_name]

            else:
                collection = self.mongo_client[database_name][collection_name]

            print(len(list(collection.find())))    

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise SpamException7(e, sys)