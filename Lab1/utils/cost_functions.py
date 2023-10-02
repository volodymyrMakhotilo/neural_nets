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
    pass

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
    pass

class MSE(Cost_Function):
    pass