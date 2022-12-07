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
    test_split_ratio: float
    failed_data_ingestion_dir: Path
    meta_data_file_path: Path
    data_ingestion_dir_name: str
    source_data_file_name: str