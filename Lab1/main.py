import numpy as np
import pandas as pd
from numpy import sqrt
from numpy.random import uniform, randn
from utils.cost_functions import Sparse_Categorical_Crossentropy, MSE, Binary_Crossentropy
from utils.optimizers import gradient_descent
from utils.activations import ReLU, Softmax, Sigmoid, Linear
from utils.metrics import accuracy_categorical, accuracy_binary
from utils.metrics import metrics_binary
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

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
        z = self.compute_linear(X)
        a = self.activation.compute(z)
        return a

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
        return np.random.randn(width, height)

class Neural_Net:
    # FIX COST AND OPT INIT
    def __init__(self, X_train, y_train, X_test, y_test, cost_function, metric=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.layers = []
        #SPECIFY COST
        self.cost_function = cost_function
        #SPECIFY OPTIMIZER
        self.optimizer = gradient_descent

    # Didn't put much thought in code
    # Batch is not coded
    def forward_propagation(self, batch_train, batch_test):
        output_train = batch_train
        output_test = batch_test
        for layer in self.layers:
            output_test = layer.forward(output_test)
            output_train = layer.forward(output_train)

        if self.metric is not None:
            print('loss_train', self.cost_function.compute(self.y_train, output_train), 'acc', self.metric(self.y_train, output_train),
                  "loss_test", self.cost_function.compute(self.y_test, output_test), 'acc', self.metric(self.y_test, output_test))
        else:
            print('loss_train', self.cost_function.compute(self.y_train, output_train), "loss_test", self.cost_function.compute(self.y_test, output_test))
        return output_train

    def backward_propagation(self, dX):
        dX = dX
        for layer in self.layers[::-1]:
            dX = layer.backward(dX)

    def cost(self, y_true, y_pred):
        pass


    def iteration(self):
        # Pass batch
        y_pred = self.forward_propagation(self.X_train, self.X_test)
        loss = self.cost(y_pred)


    def optimizer(self):
        pass

    def update_epoch(self, epoch):
        for layer in self.layers:
            layer.epoch = epoch

    #Batch
    def fit(self, epochs):
        for epoch in range(epochs):
            y_pred_train = self.forward_propagation(self.X_train, self.X_test)
            self.backward_propagation(self.cost_function.derivative(self.y_train, y_pred_train))
            self.update_epoch(epoch)

    def add(self, layer):
        self.layers.append(layer)

def main():
    data_test = pd.read_csv("data/preprocessed/boston_housing/test_boston_housing.csv")
    data_train = pd.read_csv("data/preprocessed/boston_housing/train_boston_housing.csv")
    target_label = data_train.columns[-1]
    X_test = data_test.drop(target_label, axis=1).to_numpy()
    y_test = np.expand_dims(data_test[target_label].to_numpy(), axis=-1)
    X_train = data_train.drop(target_label, axis=1).to_numpy()
    y_train = np.expand_dims(data_train[target_label].to_numpy(), axis=-1)
    #make_classification(n_samples=6, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, n_classes= 3)
    #enc = OneHotEncoder()
    #y = enc.fit_transform(np.expand_dims(y, axis=-1)).toarray()

    print(X_test.shape)
    print(y_test.shape)


    model = Neural_Net(X_train, y_train, X_test, y_test, MSE())

    hidden_layer = Layer(X_test.shape[-1], 8, ReLU())
    output_layer = Layer(hidden_layer.output_size, 1, Linear())

    model.add(hidden_layer)
    model.add(output_layer)

    model.fit(1000)


if __name__ == '__main__':
    main()