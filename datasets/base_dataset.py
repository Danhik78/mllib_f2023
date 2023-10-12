import numpy as np
from abc import ABC, abstractmethod
import random


class BaseDataset(ABC):

    def __init__(self,train_set_percent,valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass


    def _divide_into_sets(self):
        n = len(self.inputs)
        shuffled_index = np.arange(n)
        np.random.shuffle(shuffled_index)
        train_index = shuffled_index[:int(n*self.train_set_percent)]
        valid_index = shuffled_index[int(n*self.train_set_percent):int(n*(self.train_set_percent+self.valid_set_percent))]
        test_index = shuffled_index[-int(n*self.valid_set_percent):]
        self.inputs_train = self.inputs[train_index]
        self.inputs_valid = self.inputs[valid_index]
        self.inputs_test = self.inputs[test_index]
        self.targets_train = self.targets[train_index]
        self.targets_valid = self.targets[valid_index]
        self.targets_test = self.targets[test_index]
        pass
