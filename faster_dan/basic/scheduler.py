from torch.nn import Dropout, Dropout2d
import numpy as np


class DropoutScheduler:

    def __init__(self, models, function, T=1e5):
        """
        T: number of gradient updates to converge
        """

        self.teta_list = list()
        self.init_teta_list(models)
        self.function = function
        self.T = T
        self.step_num = 0

    def step(self):
        self.step(1)

    def step(self, num):
        self.step_num += num

    def init_teta_list(self, models):
        for model_name in models.keys():
            self.init_teta_list_module(models[model_name])

    def init_teta_list_module(self, module):
        for child in module.children():
            if isinstance(child, Dropout) or isinstance(child, Dropout2d):
                self.teta_list.append([child, child.p])
            else:
                self.init_teta_list_module(child)

    def update_dropout_rate(self):
        for (module, p) in self.teta_list:
            module.p = self.function(p, self.step_num, self.T)


def exponential_dropout_scheduler(dropout_rate, step, max_step):
    return dropout_rate * (1 - np.exp(-10 * step / max_step))


def linear_scheduler(init_value, end_value, step, max_step):
    return init_value + step * (end_value - init_value) / max_step