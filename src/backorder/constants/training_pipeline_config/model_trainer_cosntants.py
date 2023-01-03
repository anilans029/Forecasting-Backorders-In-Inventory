import os
from pathlib import Path

MODEL_TRAINER_ARTIFACT_DIR_NAME = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME= "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME="model.pkl"
MODEL_TRAINER_BASE_ACCURACY = 0.02
MODEL_TRAINER_MODELS_CONFIG_FILE_PATH = Path(os.path.join('config/model.yaml'))