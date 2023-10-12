from math import sin

from easydict import EasyDict
import numpy as np

from utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
#cfg.dataframe_path = "C:/Users/Danila/PycharmProjects/mllib_f2023/linear_regression_dataset.csv"
cfg.dataframe_path ="C:/Users/Danila/Downloads/linear_regression_dataset_with_inputs_as_vectors.csv"
cfg.amount_of_base_functions = 50
cfg.base_functions = [lambda x,i_=i: x[i_%3]**(i_/3) for i in range(cfg.amount_of_base_functions*3)]
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.normal_equation

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 1000

#cfg.exp_name = ''
cfg.env_path = 'C:/Users/Danila/PycharmProjects/mllib_f2023/env.env' # Путь до файла .env где будет храниться api_token.
cfg.project_name = "danilaofmadness/mllib"

