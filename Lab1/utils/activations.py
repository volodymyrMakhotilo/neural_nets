from abc import ABC, abstractmethod

import numpy as np
import math as m

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
        dx = (x >= 0).astype(np.uint)
        return dx

class Sigmoid(Activation_Function):
    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.compute(x) * (1 - self.compute(x))

class TanH(Activation_Function):
    def compute(self, x):
        return m.tanh(x)

    def derivative(self, x):
        return 1 - self.compute(x) ** 2

class LeakyReLU(Activation_Function):
    def compute(self, x):
        return np.maximum(0.01 * x, x)
    def derivative(self, x):
        if x < 0: return 0.01
        else: return 1

class Linear(Activation_Function):
    def compute(self, x):
        return x
    def derivative(self, x):
        return 1

class Softmax(Activation_Function):
    def compute(self, x):
        x = np.float64(x)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def derivative(self, x):
        return self.compute(x) * (1 - self.compute(x))


