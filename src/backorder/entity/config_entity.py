from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    
    aws_data_source_bucket_name: str
    aws_artifacts_bucket_name: str
    feature_store_dir: Path
    good_data_dir: Path
    bad_data_dir: Path
    feature_store_merged_filePath: Path
    feature_store_raw_data_dir: Path
    meta_data_file_path: Path
    data_ingestion_dir_name: str
    source_data_file_name: str

@dataclass(frozen=True)
class DataValidationConfig:

    data_validation_artifact_dir: Path
    valid_data_dir: Path
    invalid_data_dir: Path
    test_split_ratio: float
    validation_split_ratio: float
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
    
@dataclass(frozen= True)
class ModelTrainerConfig:

    trained_model_file_path: Path
    models_config_file_path: Path
    base_accuracy: float

@dataclass(frozen= True)
class ModelEvaluationConfig:

    changed_accuracy_threshold: float
    model_registry_bucket_latest_model_key : str
    model_registry_latest_model_dir: str
    model_registry_bucket_name: str
    best_model_obj_file_name: str
    best_model_artifact_file_path: Path
    accepted_model_artifact_file_path: Path

@dataclass(frozen= True)
class ModelPusherConfig:
    model_registry_bucket_name: str
    model_registry_latest_model_dir_key: str
    model_registry_latest_model_obj_key: str
    model_registry_previous_model_obj_key: str
    model_registry_previous_model_dir: str
    