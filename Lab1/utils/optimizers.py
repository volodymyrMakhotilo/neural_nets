import numpy as np
from numpy.linalg import norm
from abc import ABC, abstractmethod

e = 1E-3


class Optimizer(ABC):
    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def update(self, W, b, dW, db, epoch, layer_num):
        pass

    @abstractmethod
    def compute_change(self, dx, epoch, dxs):
        pass

    def check_betta(self, betta):
        if betta < 0 or betta > 1:
            raise Exception("Betta should be in [0, 1] range")

    def init_grad_dict(self, dWs, dbs, layers_num):
        for layer_num in range(layers_num):
            dWs[layer_num] = 0
            dbs[layer_num] = 0
        return dWs, dbs

    def clip(self, grad):
        grad[grad == 0] = e
        return np.clip(grad, -1, 1)


class GradientDescent(Optimizer):
    def update(self, W, b, dW, db, epoch, **kwargs):
        W = W - self.compute_change(dW, epoch)
        b = b - self.compute_change(db, epoch)
        return W, b

    def compute_change(self, dx, epoch, **kwargs):
        grad = 0.99 ** epoch * self.alpha * dx
        return self.clip(grad)


class AdaGrad(Optimizer):
    def __init__(self, alpha, layers_num):
        super().__init__(alpha)
        self.dWs = {}
        self.dbs = {}
        for layer_num in range(layers_num):
            self.dWs[layer_num] = 0
            self.dbs[layer_num] = 0

    def update(self, W, b, dW, db, epoch, layer_num):
        self.dWs[layer_num] += dW ** 2
        self.dbs[layer_num] += db ** 2

        W = W - self.compute_change(dW, epoch, self.dWs[layer_num])
        b = b - self.compute_change(db, epoch, self.dbs[layer_num])
        return W, b

    def compute_change(self, dx, epoch, dxs):
        grad = self.alpha / (np.sqrt(dxs) + e) * dx
        return self.clip(grad)


class RMSprop(Optimizer):
    def __init__(self, alpha, betta, layers_num):
        super().__init__(alpha)
        self.dWs = {}
        self.dbs = {}
        for layer_num in range(layers_num):
            self.dWs[layer_num] = 0
            self.dbs[layer_num] = 0
        self.check_betta(betta)
        self.betta = betta

    def update(self, W, b, dW, db, epoch, layer_num):
        self.dWs[layer_num] = self.betta * self.dWs[layer_num] + (1 - self.betta) * dW ** 2
        self.dbs[layer_num] = self.betta * self.dbs[layer_num] + (1 - self.betta) * db ** 2

        W = W - self.compute_change(dW, epoch, self.dWs[layer_num])
        b = b - self.compute_change(db, epoch, self.dbs[layer_num])
        return W, b

    def compute_change(self, dx, epoch, dxs):
        grad = self.alpha / (np.sqrt(dxs) + e) * dx
        return self.clip(grad)


class Adam(Optimizer):
    def __init__(self, alpha, betta_m, betta_v, layers_num):
        super().__init__(alpha)
        self.dWms, self.dbms = self.init_grad_dict({}, {}, layers_num)
        self.dWvs, self.dbvs = self.init_grad_dict({}, {}, layers_num)
        self.check_betta(betta_m)
        self.check_betta(betta_v)
        self.betta_m = betta_m
        self.betta_v = betta_v

    def update(self, W, b, dW, db, epoch, layer_num):
        self.dWvs[layer_num] = self.betta_v * self.dWvs[layer_num] + (1 - self.betta_v) * dW ** 2
        self.dbvs[layer_num] = self.betta_v * self.dbvs[layer_num] + (1 - self.betta_v) * db ** 2
        self.dWms[layer_num] = self.betta_m * self.dWms[layer_num] + (1 - self.betta_m) * dW
        self.dbms[layer_num] = self.betta_m * self.dbms[layer_num] + (1 - self.betta_m) * db

        dWvs_corr = self.dWvs[layer_num] / (1 - self.betta_v ** epoch)
        dWms_corr = self.dWms[layer_num] / (1 - self.betta_m ** epoch)
        dbvs_corr = self.dbvs[layer_num] / (1 - self.betta_v ** epoch)
        dbms_corr = self.dbms[layer_num] / (1 - self.betta_m ** epoch)

        W = W - self.compute_change(dWms_corr, epoch, dWvs_corr)
        b = b - self.compute_change(dbms_corr, epoch, dbvs_corr)

        return W, b

    def compute_change(self, dx, epoch, dxs):
        grad = self.alpha / (np.sqrt(dxs) + e) * dx
        return self.clip(grad)
