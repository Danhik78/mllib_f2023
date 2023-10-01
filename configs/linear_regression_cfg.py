from easydict import EasyDict
import numpy as np
cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = "C:/Users/Danila/PycharmProjects/mllib_f2023/linear_regression_dataset.csv"

cfg.base_functions = []

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 100

#cfg.exp_name = ''
cfg.env_path = '' # Путь до файла .env где будет храниться api_token.
cfg.project_name = ''

