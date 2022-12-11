from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    
    aws_data_source_bucket_name: str
    aws_artifacts_bucket_name: str
    ingested_data_dir: Path
    feature_store_dir: Path
    feature_store_merged_filePath: Path
    feature_store_raw_data_dir: Path
    train_file_path: Path
    test_file_path: Path
    validation_file_path: Path
    test_split_ratio: float
    validation_split_ratio: float
    failed_data_ingestion_dir: Path
    meta_data_file_path: Path
    data_ingestion_dir_name: str
    source_data_file_name: str

@dataclass(frozen=True)
class DataValidationConfig:

    data_validation_artifact_dir: Path
    valid_data_dir: Path
    invalid_data_dir: Path
    valid_train_file_path: Path
    valid_validation_file_path: Path
    invalid_train_file_path: Path
    valid_test_file_path: Path
    invalid_test_file_path: Path
    invalid_validation_file_path: Path
    ks_2_samp_test_threshold: float
    drift_report_file_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:

    data_transformation_artifact_dir: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    transformed_valid_file_path: Path
    preprocessor_obj_file_path: Path
    