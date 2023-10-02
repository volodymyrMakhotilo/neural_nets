from abc import ABC, abstractmethod

import numpy as np


class Activation_Function(ABC):
    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def derivative(self):
        pass


class ReLU(Activation_Function):
    def compute(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        dx = x * (x >= 0).astype(np.uint)
        return dx

class Sigmoid(Activation_Function):
    pass

class TanH(Activation_Function):
    pass

class LeakyReLU(Activation_Function):
    pass

class Linear(Activation_Function):
    pass

class Softmax(Activation_Function):
    def compute(self, x):
        x = np.float64(x)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def derivative(self, x):
        return self.compute(x) * (1 - self.compute(x))


