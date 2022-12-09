from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:

    train_file_path: Path
    test_file_path: Path
    feature_file_path: Path
@dataclass(frozen=True)
class DataValidationArtifact:

    validation_status: bool
    valid_train_filepath: Path
    valid_test_filepath: Path
    Invalid_train_filepath: Path
    Invalid_test_filepath: Path
    drift_report_filepath: Path

@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_obj_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path