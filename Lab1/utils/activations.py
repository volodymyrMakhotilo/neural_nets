from abc import ABC, abstractmethod

import numpy as np
import math as m

class Activation_Function(ABC):
    @abstractmethod
    def compute(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class ReLU(Activation_Function):
    def compute(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        dx = (x >= 0).astype(np.uint)
        return dx

class Sigmoid(Activation_Function):
    def positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def negative_sigmoid(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)

    def compute(self, x):
        positive = x >= 0
        negative = ~positive
        result = np.empty_like(x, dtype=np.float64)
        result[positive] = self.positive_sigmoid(x[positive])
        result[negative] = self.negative_sigmoid(x[negative])
        return result

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
        return np.ones_like(x)

class Softmax(Activation_Function):
    def compute(self, x):
        # Technic to prevent overflow https://stats.stackexchange.com/questions/304758/softmax-overflow
        # Technic results in numeric underflow due to float number arithmetics
        m = np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x - m)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def derivative(self, x):
        return self.compute(x) * (1 - self.compute(x))


print()

