# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import numpy as np
from datasets.linear_regression_dataset import LinRegDataset
from configs.linear_regression_cfg import cfg
from models.linear_regression_model import LinearRegression
from utils.enums import TrainType
from utils.metrics import MSE
from utils.visualisation import Visualisation
import numpy as np
from utils.common_functions import read_dataframe_file

if __name__ == '__main__':
    data = LinRegDataset(cfg,['x_0','x_1','x_2'])
    cfg.train_type= TrainType.gradient_descent
    lin_model = LinearRegression(cfg.base_functions,0.1)
    lin_model.train(data.inputs_train, data.targets_train)
    lin_model.neptune_logger.log_final_val_mse(MSE(lin_model(data.inputs_valid),data.targets_valid))
    print(f"trainMSE = {MSE(lin_model(data.inputs_train),data.targets_train)}")
    print(f"validMSE = {MSE(lin_model(data.inputs_valid),data.targets_valid)}")