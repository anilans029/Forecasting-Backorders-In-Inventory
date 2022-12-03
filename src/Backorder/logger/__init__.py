import logging
from DateTime.DateTime import datetime
import os
from from_root import from_root

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_DIR = os.path.join(from_root(),"logs")

os.makedirs(LOG_DIR, exist_ok= True)
LOG_FILE_PATH = os.path.join(from_root(),LOG_DIR,LOG_FILE)

LOG_FORMAT = "[%(asctime)s]- %(lineno)d -  %(name)s - %(levelname)s => %(message)s"

logging.basicConfig(filename=LOG_FILE_PATH, filemode="w",level= logging.INFO, format= LOG_FORMAT)