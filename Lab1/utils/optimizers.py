import numpy as np
from numpy.linalg import norm
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def update(self, W, b, dW, db, epoch, **kwargs):
        pass

    @abstractmethod
    def compute_change(self, dx, epoch, **kwargs):
        pass


class GradientDescent(Optimizer):
    def update(self, W, b, dW, db, epoch, **kwargs):
        W = W - self.compute_change(dW, epoch)
        b = b - self.compute_change(db, epoch)
        return W, b

    def compute_change(self, dx, epoch):
        return np.clip(0.99 ** epoch * self.alpha * dx, -1, 1)


class AdaGrad(Optimizer):
    def __init__(self, alpha, layers_num):
        super().__init__(alpha)
        self.dWs = {}
        self.dbs = {}
        for layer_num in range(layers_num):
            self.dWs[layer_num] = np.array([[[]]])
            self.dbs[layer_num] = np.array([[[]]])

    def update(self, W, b, dW, db, epoch, layer_num):
        self.dWs[layer_num] = np.reshape(self.dWs[layer_num], [dW.shape[0], dW.shape[1], -1])
        self.dbs[layer_num] = np.reshape(self.dbs[layer_num], [db.shape[0], db.shape[1], -1])

        self.dWs[layer_num] = np.append(self.dWs[layer_num], np.expand_dims(dW, axis=-1), axis=-1)
        self.dbs[layer_num] = np.append(self.dbs[layer_num], np.expand_dims(db, axis=-1), axis=-1)

        W = W - self.compute_change(dW, epoch, self.dWs[layer_num])
        b = b - self.compute_change(db, epoch, self.dbs[layer_num])

        return W, b

    def compute_change(self, dx, epoch, dxs):
        return self.alpha / (norm(dxs, ord=2, axis=-1) + 1E-10) * dx
