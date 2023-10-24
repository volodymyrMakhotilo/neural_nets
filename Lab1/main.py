import numpy as np
import pandas as pd
from numpy import sqrt
from numpy.random import uniform, randn
from utils.cost_functions import Sparse_Categorical_Crossentropy, MSE
from utils.optimizers import gradient_descent
from utils.activations import ReLU, Softmax, Sigmoid, Linear
from utils.metrics import accuracy_categorical
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

class Layer:
    # Shapes should be compatible for matrix multiplication
    def __init__(self, input_size, output_size, activation):
        self.batch_size = 0
        self.input_size = input_size
        self.output_size = output_size
        self.epoch = 0
        self.bias = self.weights_init(1, output_size)
        self.weights = self.weights_init(output_size, input_size)
        self.activation = activation
        self.cached_input = np.NaN

    def forward(self, X):
        self.batch_size = X.shape[0]
        self.cached_input = X
        return self.activation.compute(self.compute_linear(X))

    def compute_linear(self, X):
        return np.matmul(X, self.weights.T) + self.bias

    def update(self, grad):
        dW = np.matmul(grad.T, self.cached_input) / self.batch_size
        db = np.mean(grad, axis=0)
        self.weights = gradient_descent(self.weights, dW, self.epoch)
        self.bias = gradient_descent(self.bias, db, self.epoch)

    def backward(self, dX):
      grad = dX * self.activation.derivative(self.compute_linear(self.cached_input))
      ouput = np.matmul(grad, self.weights)
      self.update(grad)

      return ouput

    # Experiment!
    def weights_init(self, width, height):
        lower, upper = -(sqrt(6.0) / sqrt(self.input_size + self.output_size)), (sqrt(6.0) / sqrt(self.input_size + self.batch_size))
        return lower + np.random.randn(width, height) * (upper - lower)

class Neural_Net:
    # FIX COST AND OPT INIT
    def __init__(self, X_train, y_train, cost_function):
        self.X_train = X_train
        self.y_train = y_train
        self.layers = []
        #SPECIFY COST
        self.cost_function = cost_function
        #SPECIFY OPTIMIZER
        self.optimizer = gradient_descent

    # Didn't put much thought in code
    # Batch is not coded
    def forward_propagation(self, batch):
        output = batch
        for layer in self.layers:
            output = layer.forward(output)
        print('loss', self.cost_function.compute(self.y_train, output), 'acc', accuracy_categorical(self.y_train, output))
        return output

    def backward_propagation(self, dX):
        dX = dX
        for layer in self.layers[::-1]:
            dX = layer.backward(dX)

    def cost(self, y_true, y_pred):
        pass


    def iteration(self):
        # Pass batch
        y_pred = self.forward_propagation(self.X_train)
        loss = self.cost(y_pred)


    def optimizer(self):
        pass

    def update_epoch(self, epoch):
        for layer in self.layers:
            layer.epoch = epoch

    #Batch
    def fit(self, epochs):
        for epoch in range(epochs):
            y_pred = self.forward_propagation(self.X_train)
            self.backward_propagation(self.cost_function.derivative(self.y_train, y_pred))
            self.update_epoch(epoch)

    def add(self, layer):
        self.layers.append(layer)

def main():
    data = pd.read_csv("data/preprocessed/boston_housing/train_boston_housing.csv")
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    #make_classification(n_samples=6, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, n_classes= 3)
    enc = OneHotEncoder()
    y = enc.fit_transform(np.expand_dims(y, axis=-1)).toarray()


    model = Neural_Net(X, y, MSE())

    hidden_layer = Layer(X.shape[-1], 5, ReLU())
    output_layer = Layer(hidden_layer.output_size, 3, Linear())

    model.add(hidden_layer)
    model.add(output_layer)

    model.fit(50)


if __name__ == '__main__':
    main()