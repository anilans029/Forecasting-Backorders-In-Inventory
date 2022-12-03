import os, sys
from backorder.exception import BackorderException
from backorder.logger import logging

class S3Sync:

    def sync_s3_to_folder(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            logging.info(f"syncing folder from s3")
            os.system(command= command)

        except Exception as e:
            logging.info(BackorderException(e,sys))
            
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            logging.info(f"syncing folder to s3")
            os.system(command)
            
        
        except Exception as e:
            logging.info(BackorderException(e,sys))