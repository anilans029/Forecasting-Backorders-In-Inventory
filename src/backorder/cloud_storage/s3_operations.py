import os, sys
from backorder.exception import BackorderException
from backorder.logger import logging
import boto3

class S3Operations:
    
    def __init__(self):
        try:
        
            self.s3_client = boto3.client("s3",
                            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
                            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
                            )
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)
            
    def sync_s3_to_folder(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            logging.info(f"syncing folder from s3")
            os.system(command= command)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            logging.info(f"syncing folder to s3")
            os.system(command)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def create_s3_bucket(self, bucket_name: str):
        try:
            response = self.s3_client.create_bucket(Bucket= bucket_name,
                                                CreateBucketConfiguration= {
                                                    "LocationConstraint": os.getenv("AWS_DEFAULT_REGION")
                                                })

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def upload_file_to_s3(self,src_path, bucket_name, object_destination_path):
        try:
            response = self.s3_client.upload_file(src_path,
                                            bucket_name,
                                            object_destination_path)

        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def download_file_from_s3(self,bucket_name,object_path,download_file_path):
        try:
            # s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME for the downloading object')
            self.s3_client.download_file(
                                    bucket_name,
                                    object_path,
                                    download_file_path                                    
                                    )
        except Exception as e:
            logging.info(BackorderException(e, sys))
            raise BackorderException(e,sys)

    def list_all_buckets_in_s3(self):
        try:
            buckets = []
            response = self.s3_client.list_buckets()
            for i in response['Buckets']:
                buckets.append(i["Name"])
            return buckets
        
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def list_all_objects_in_s3Bucket(self, bucket_name):
        try:
            buckets = self.list_all_buckets_in_s3()
            objects= []
            if bucket_name in buckets:
                response = self.s3_client.list_objects(Bucket = bucket_name)
                for content in response.get('Contents'):
                    objects.append(content.get("Key"))
                return objects
            else:
                logging.info(BackorderException(e,sys))
                raise Exception('No such bucket exists in s3')

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)



     




     
