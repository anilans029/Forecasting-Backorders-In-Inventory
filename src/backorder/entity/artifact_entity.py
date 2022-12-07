from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:

    train_file_path: Path
    test_file_path: Path
    feature_file_path: Path