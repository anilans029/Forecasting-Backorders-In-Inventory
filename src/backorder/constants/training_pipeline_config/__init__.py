from backorder.constants.training_pipeline_config.data_ingestion_constants import *
from backorder.constants.training_pipeline_config.data_validation_constants  import *
from backorder.constants.training_pipeline_config.data_transformation_constants import *
from backorder.constants.training_pipeline_config.model_trainer_cosntants import *
from backorder.constants.training_pipeline_config.model_evaluation_constants import *
from datetime import datetime
### common constants for all the components in training pipeline
ARTIFACT_DIR = "artifact"
TIMESTAMP = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
