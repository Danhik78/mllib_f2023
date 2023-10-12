# TODO:
#  1) Load the dataset using pandas read_csv function.
#  2) Split the dataset into training, validation, and test sets.
#  Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  Use class from datasets.linear_regression_dataset.py
#  3) Define hyperparameters space
#  4) Use loop where you randomly choose hypeparameter from space and train model
#  5) Create experiment name using code from logging_example.py
#  6) Initialize the Linear Regression model using the provided `LinearRegression` class
#  7) Log hyperparameters to neptune
#  8) Train the model using the training data and gradient descent,
#  log MSE and cost function on validation and trainig sets
#  9) Log final mse on validation set after trainig
#  10) Save model if it is showing best mse on validation set

import sys

from datasets.linear_regression_dataset import LinRegDataset
from configs.linear_regression_cfg import cfg
from models.linear_regression_model import LinearRegression
from utils.enums import TrainType
from utils.metrics import MSE
from utils import base_funs as fs
from logs import Logger

data = LinRegDataset(cfg,['x_0','x_1','x_2'])
cfg.train_type = TrainType.gradient_descent
best_mse = sys.float_info.max
best_mse_model = None
for lr in [0.1,0.01,0.2]:
    for reg_reit in [0.0,0.001,0.01]:
        for funs in [fs.poly_funs(3,3),fs.poly_funs(3,8),fs.poly_funs(3,100),fs.sin_funs(3,3)]:

            model = LinearRegression(funs,lr,reg_reit)
            logger = model.create_logger(data.inputs_valid,data.targets_valid)
            model.train(data.inputs_train,data.targets_train,logger)
            mse = MSE(model(data.inputs_valid),data.targets_valid)
            if mse < best_mse:
                best_mse = mse
                best_mse_model = model
print(best_mse)
best_mse_model.save('best_model.pkl')
