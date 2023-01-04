from backorder.utils import read_yaml, write_yaml
from backorder.ml.model.esitmator import TargetValueMapping
import importlib
from backorder.pipeline.training_pipeline import TrainingPipeline
from backorder.pipeline.prediciton_pipeline import PredictionPipeline
import pandas as pd
from backorder.logger import logging


# training_pipeline = TrainingPipeline()
# prediction_pipeline = PredictionPipeline()
# # model_config = read_yaml("config\model.yaml")

# # print(type(model_config["model_selection"]))

# # data_validation = DataValidation()

# # dic1 = {"numerical_features":{'sales_6_month': 'float64',
# #  'sales_9_month': 'float64',
# #  'min_bank': 'float64',
# #  'potential_issue': 'object',
# #  'pieces_past_due': 'float64',
# #  'perf_6_month_avg': 'float64',
# #  'perf_12_month_avg': 'float64'}}

# # write_yaml(content=dic1, file_path="demo/sample.yaml")

# # print(TargetValueMapping().to_dict())

# df = pd.read_csv("sample_dataset\\backorders_data_sample.csv")
# df =df.sample(n=1)
# # df.drop(columns=["sku","went_on_backorder"],inplace=True)
# # pred_df =  prediction_pipeline.single_instance_predict(df)
# # logging.info(f"{pred_df.value_counts()}")
# prediction_pipeline.start_single_instance_prediction(dataframe=df)

# # prediction_pipeline.start_batch_prediction()


from backorder.pipeline.training_pipeline import TrainingPipeline
training = TrainingPipeline()
training.start()