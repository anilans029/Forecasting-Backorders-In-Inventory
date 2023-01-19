from pathlib import Path
import os

SCHEMA_FILE_TOTAL_NO_COLUMNS = "total_no_of_columns"
SCHEMA_FILE_TOTAL_NUMERICAL_COLUMNS = "total_numerical_columns"
SCHEMA_FILE_TOTAL_CATEGORICAL_COLUMNS = "total_categorical_columns"
SCHEMA_FILE_ALL_COLUMNS="All_columns"
SCHEMA_FILE_NUMERICAL_FEATURES_DATA_TYPES = "Numerical_features_dataypes"
SCHEMA_FILE_CATEGORICAL_FEATURES_DATA_TYPES = "Categorical_features_datatypes"
SCHEMA_FILE_TARGET_COLUMNS_DATA_TYPE = "Target_column"
SCHEMA_FILE_TARGET_COLUMN_NAME= "Target_column_Name"
SCHEMA_FILE_UNWANTED_COLUMNS = "Unwanted_columns"
SCHEMA_FILE_PATH = Path(os.path.join("config","schema.yaml"))
SCHEMA_FILE_EXTRACTED_FEATURES= 'Extracted_features'