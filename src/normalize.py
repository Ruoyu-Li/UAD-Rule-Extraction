import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from global_var import *


class Normalizer():
    def __init__(self):
        pass

    def fit(self, x):
        self.std_scaler = StandardScaler()
        x_tmp = self.std_scaler.fit_transform(x)
        self.x_max, self.x_min = np.max(x_tmp, axis=0), np.min(x_tmp, axis=0)
        self.r = 1

    def normalize(self, x):
        if type(x) == torch.Tensor:
            x_tmp = x.numpy()
        else:
            x_tmp = x
        x_tmp = self.std_scaler.transform(x_tmp)
        x_tmp = 2 * self.r * ((x_tmp - self.x_min) / (self.x_max - self.x_min) - 1 / 2)
        x_tmp[np.isnan(x_tmp)] = 0
        if type(x) == torch.Tensor:
            return torch.Tensor(x_tmp)
        else:
            return x_tmp

    def denormalize(self, x):
        if type(x) == torch.Tensor:
            x_tmp = x.numpy()
        else:
            x_tmp = x
        x_tmp = (self.x_max - self.x_min) * (x_tmp / (2 * self.r) + 1 / 2) + self.x_min
        x_tmp = self.std_scaler.inverse_transform(x_tmp)
        if type(x) == torch.Tensor:
            return torch.Tensor(x_tmp)
        else:
            return x_tmp


class mm_Normalizer():
    def __init__(self):
        pass

    def fit(self, x):
        self.mm_scaler = MinMaxScaler()
        self.mm_scaler.fit(x)
        self.r = 0.8

    def normalize(self, x):
        if type(x) == torch.Tensor:
            x_tmp = x.numpy()
        else:
            x_tmp = x
        x_tmp = self.mm_scaler.transform(x_tmp)
        x_tmp = self.r * x_tmp
        if type(x) == torch.Tensor:
            return torch.Tensor(x_tmp)
        else:
            return x_tmp

class std_Normalizer():
    def __init__(self):
        pass

    def fit(self, x):
        self.std_scaler = StandardScaler()
        self.std_scaler.fit(x)

    def normalize(self, x):
        if type(x) == torch.Tensor:
            x_tmp = x.numpy()
        else:
            x_tmp = x
        x_tmp = self.std_scaler.transform(x_tmp)
        if type(x) == torch.Tensor:
            return torch.Tensor(x_tmp)
        else:
            return x_tmp