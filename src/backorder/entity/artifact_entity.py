from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:

    train_file_path: Path
    test_file_path: Path
    validation_file_path: Path
    feature_file_path: Path
@dataclass(frozen=True)
class DataValidationArtifact:

    validation_status: bool
    valid_train_filepath: Path
    valid_test_filepath: Path
    valid_validation_file_path: Path
    Invalid_train_filepath: Path
    Invalid_test_filepath: Path
    Invalid_validation_file_path: Path
    drift_report_filepath: Path

@dataclass(frozen=True)
class DataTransformationArtifact:
    is_Transformed: bool
    transformed_obj_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    transformed_valid_file_path: Path
    message: str

@dataclass(frozen=True)
class ModelTrainerArtiact:
    trained_model_file_path: Path
    is_model_found: bool
    train_f1_score: float
    validation_f1_score: float
    train_precision: float
    train_recall: float
    validation_precision: float
    validation_recall: float
    

@dataclass(frozen= True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    best_model_path: str
    accepted_model_path: str
    best_model_metric_artifact: dict
