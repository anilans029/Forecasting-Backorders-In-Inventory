import os, sys
from backorder.exception import BackorderException
from backorder.logger import logging
import boto3
from io import StringIO
import pandas as pd


class S3Operations:
    
    def __init__(self):
        try:
            self.s3_resource = boto3.resource("s3",
                            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
                            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
                            )
        
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

    def is_bucket_available_in_s3(self,bucket_name:str)->bool:
        try:
            buckets_list = self.list_all_buckets_in_s3()
            if bucket_name in buckets_list:
                return True
            else:
                return False
        
        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def list_all_objects_in_s3Bucket(self, bucket_name, prefix=None):
        try:
            objects= []
            if self.is_bucket_available_in_s3(bucket_name=bucket_name):
                if prefix is None:
                    response = self.s3_client.list_objects(Bucket = bucket_name)
                else:
                    response = self.s3_client.list_objects(Bucket = bucket_name, Prefix= prefix)

                if response.get("Contents")!= None:
                    for content in response.get('Contents'):
                        objects.append(content.get("Key"))
                    return objects
                else:
                    return None
            else:
                raise Exception('No such bucket exists in s3')

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e,sys)

    def is_s3_key_path_available(self, bucket_name, s3_key) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [
                            file_object for file_object in bucket.objects.filter(Prefix=s3_key)
                           ]
            if len(file_objects) > 0:
                return True
            else:
                return False

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def get_bucket(self, bucket_name: str):
    
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

     

    def read_csv_file_from_s3(self, bucket_name:str, key:str)->pd.DataFrame:
        try:
            if self.is_s3_key_path_available(bucket_name=bucket_name,s3_key=key):
                resp = self.s3_client.get_object(Bucket= bucket_name,
                                                Key = key
                                                )

                csv_string = resp["Body"].read().decode("utf-8")
                df = pd.read_csv(StringIO(csv_string))
                return df
            else:
                raise Exception("key path not available in s3 bucket")

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def save_dataframe_as_csv_s3(self, bucket_name: str, key:str,dataframe:pd.DataFrame):
        try:
            csv_buf = StringIO()
            dataframe.to_csv(csv_buf, header=True,index=False)
            csv_buf.seek(0)
            self.s3_client.put_object(Bucket=bucket_name,Key=key, Body = csv_buf.getvalue()  )

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)

    def save_object_to_s3(self,object_body, bucket, key):
        
        try:

            self.s3_client.put_object(Body=object_body, Bucket=bucket, Key=key)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)


    def save_artifacts_to_s3(bucket_name, key):
        try:

            self.s3_client.put_object(Body=object_body, Bucket=bucket, Key=key)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            raise BackorderException(e, sys)
