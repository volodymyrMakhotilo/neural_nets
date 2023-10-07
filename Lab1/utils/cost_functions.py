from abc import ABC, abstractmethod
import numpy as np

class Cost_Function(ABC):
    @abstractmethod
    def compute(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class Binary_Crossentropy(Cost_Function):
    def compute(self, y_true, y_pred):
        epsilon = 1e-15 # Small constant to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip values to avoid numerical instability
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Log N is under question
# y_true should be sparse
class Sparse_Categorical_Crossentropy(Cost_Function):
    def compute(self, y_true, y_pred):
        #class_num = y_true.shape[-1]
        loss = np.mean(np.sum(-(y_true * np.log(y_pred)), axis=1, keepdims=True))
        return loss

    def derivative(self, y_true, y_pred):
        return -(y_true / y_pred)


class MAE(Cost_Function):
    def compute(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        if y_pred < y_true:
            return -1.0
        elif y_pred > y_true:
            return 1.0
        else:
            # Return any value between -1 and 1 for the subderivative at zero
            return 0.0

class MSE(Cost_Function):
    def compute(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)