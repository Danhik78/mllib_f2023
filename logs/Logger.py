import neptune
from dotenv import load_dotenv
import os
from typing import Union, List

class Logger():
    def __init__(self, env_path, project, experiment_name=None):

        load_dotenv(env_path)
        self.run = neptune.init_run(
            project=project,
            api_token=os.environ['api_token'],
            name=experiment_name
        )



    def set_valid(self,inputs,targets):
        self.inputs = inputs
        self.targets = targets
    def log_hyperparameters(self, params: dict):
        # сохранение гиперпараметов модели
        for param, value in params.items():
            self.run[f'hyperparameters/{param}'] = value

    def save_param(self, type_set, metric_name: Union[List[str], str], metric_value: Union[List[float], float]):
        if isinstance(metric_name, List):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].append(p_v)
        else:
            self.run[f"{type_set}/{metric_name}"].append(metric_value)

    def log_final_val_mse(self, mse_value: float):
        # сохранение финальное значение mse на валидационной выборке
        self.run['final_metrics/validation_mse'] = mse_value




