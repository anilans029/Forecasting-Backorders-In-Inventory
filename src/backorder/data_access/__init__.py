from backorder.config.mongo_client import MongodbClient
from backorder.constants.mongoDB import ARTIFACTS_COLLECTION_NAME
from backorder.constants.training_pipeline_config import TIMESTAMP


class MongodbOperations:

    def __init__(self):
        self.client = MongodbClient()
        self.collection_name = ARTIFACTS_COLLECTION_NAME
        self.collection = self.client.database[self.collection_name]
        self.timestamp = TIMESTAMP

    def save_artifact(self, artifact):

        artifact["timestamp"] = TIMESTAMP 
        self.collection.insert_one(artifact)

