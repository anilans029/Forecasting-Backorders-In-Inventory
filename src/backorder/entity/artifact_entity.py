from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:

    training_phase: str
    feature_file_path: Path
    good_data_files: dict
    bad_data_files: dict
@dataclass(frozen=True)
class DataValidationArtifact:

    training_phase: str
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

    training_phase:str
    is_Transformed: bool
    transformed_obj_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    transformed_valid_file_path: Path
    message: str

@dataclass(frozen=True)
class ModelTrainerArtiact:

    training_phase: str
    trained_model_file_path:Path
    is_model_found:bool
    train_recall: float
    test_recall:float
    train_f1_each_class_scores:list
    test_f1_each_class_scores:list
    train_precision :float
    test_precision :float
    train_macro_F1 :float
    test_macro_F1 :float
    train_roc_auc :float
    test_roc_auc :float
    pr_auc_train :float
    pr_auc_test :float


@dataclass(frozen= True)
class ModelEvaluationArtifact:

    training_phase: str
    is_model_accepted: bool
    changed_pr_auc: float
    best_model_path: str
    accepted_model_path: str
    best_model_metric_artifact: dict
@dataclass(frozen=True)
class ModelPusherArtifact:

    training_phase: str
    is_model_pushed: bool
    bucket_name: str
    latest_model_obj_key: str