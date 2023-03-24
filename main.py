# from Spam.config.mongo_db_connection import MongoDBClient

# from Spam.constants import database
# COLLECTION_NAME = database.SPAM_COLLECTION_NAME

# mongo_client = MongoDBClient()
# collection = mongo_client.database[COLLECTION_NAME]
# print(list(collection.find()))



from Spam.pipeline.training import TrainPipeline
x = TrainPipeline()
x.run_pipeline()