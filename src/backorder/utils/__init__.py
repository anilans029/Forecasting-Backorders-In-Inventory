
from pathlib import Path
import yaml
import os, sys
from backorder.exception import BackorderException
from backorder.logger import logging
import shutil
import dill
import numpy as np



def copy_file(src_file: Path, destination: Path):
    os.makedirs(os.path.dirname(destination), exist_ok= True)
    shutil.copy(src= src_file, dst= destination)


def read_yaml(filepath: Path):
    """
    The read_yaml function reads a yaml file and returns the contents as a dictionary.
       
    
    Args:
        filepath:Path: Pass in the filepath of the yaml file that is being read
    
    Returns:
        A dictionary
    
    """
    try:
        with open(filepath, "r") as yaml_file:
            if os.path.exists(filepath):
                logging.info(f"reading the yaml file present at : {filepath}")
                yaml_content = yaml.safe_load(yaml_file)
                return yaml_content
            else:
                logging.info('yamll file doesnt exist at : {filepath}')
    except Exception as e:
        logging.info(BackorderException(e,sys))
        raise BackorderException(e,sys)


def write_yaml(file_path:Path, content: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, "w") as yaml_file:
            if content is not None:
                yaml.dump(content, yaml_file)

    except Exception as e:
        logging.info(BackorderException(e, sys))
        raise BackorderException(e,sys)


def create_directories(list_of_directories: list):
    """
    The create_directories function creates a list of directories.
       The function takes in a list of directories as an argument and attempts to create each directory in the list.
       If one or more of the directories already exist, no error is raised.
    
    Args:
        list_of_directories:list: Pass a list of directories that will be created
    
    
    """
    try:
        for dir in list_of_directories:
            os.makedirs(dir, exist_ok= True)

    except Exception as e:
        raise e


def save_numpy_array_data(file_path: Path, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise BackorderException(e, sys) from e


def load_numpy_array_data(file_path: Path)->np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise BackorderException(e, sys) from e


def save_object(file_path: Path, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise BackorderException(e, sys) from e


def load_object(file_path: Path)-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"file not found at the given location: { file_path }")
        with open(file= file_path, mode="rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logging.info(BackorderException(e, sys))

def read_model_byte_code(byte_code_str: str):
    try:
        model_object = dill.loads(byte_code_str)
        return model_object
    except Exception as e:
        logging.info(BackorderException(e, sys))

def read_byte_coded_yaml_file(byte_code_yaml: str):
    try:
        yaml_dict = yaml.safe_load(byte_code_yaml)
        return yaml_dict
    
    except Exception as e:
        logging.info(BackorderException(e, sys))