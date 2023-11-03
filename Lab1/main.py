import numpy as np
import pandas as pd
from numpy import sqrt
from numpy.random import shuffle
from numpy.random import uniform, randn
from utils.cost_functions import Sparse_Categorical_Crossentropy, MSE, Binary_Crossentropy
from utils.optimizers import *
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

    def get_param_grad(self, grad):
        dW = np.matmul(grad.T, self.cached_input) / self.batch_size
        db = np.mean(grad, axis=0, keepdims=True)
        return dW, db

    def backward(self, dX):
        grad = dX * self.activation.derivative(self.compute_linear(self.cached_input))
        output = np.matmul(grad, self.weights)
        dW, db = self.get_param_grad(grad)

        return output, dW, db

    # Experiment!
    def weights_init(self, width, height):
        lower, upper = -(sqrt(6.0) / sqrt(self.input_size + self.output_size)), (
                sqrt(6.0) / sqrt(self.input_size + self.batch_size))
        return np.random.randn(width, height)


class Neural_Net:
    # FIX COST AND OPT INIT
    def __init__(self, X_train, y_train, X_test, y_test, metric, cost_function, optimizer: Optimizer, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.layers = []
        # SPECIFY COST
        self.cost_function = cost_function
        # SPECIFY OPTIMIZER
        self.optimizer = optimizer

        self.batch_size = batch_size

    # Didn't put much thought in code
    # Batch is not coded
    def forward_propagation(self, batch):
        output = batch
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def get_batch_generator(self, X, y, batch_size):
        batches_num = np.ceil(X.shape[0] / batch_size)
        batches_indexes = np.arange(batches_num).astype(np.int64)
        # Inplace operation for some reason
        shuffle(batches_indexes)
        for batch_index in batches_indexes:
            yield X[batch_index:batch_index + 1, :], y[batch_index:batch_index + 1, :]

    def backward_propagation(self, grad, epoch):
        grad = grad
        layers_num = len(self.layers) - 1
        for iter_, layer in enumerate(self.layers[::-1]):
            grad, dW, db = layer.backward(grad)
            layer.weights, layer.bias = self.optimizer.update(layer.weights, layer.bias, dW, db, epoch, layer_num=layers_num-iter_)

    def cost(self, y_true, y_pred):
        pass

    def optimizer(self):
        pass

    def update_epoch(self, epoch):
        for layer in self.layers:
            layer.epoch = epoch

    def training_iteration(self, batch_gen, epoch):
        for batch_x, batch_y in batch_gen:
            y_pred_train = self.forward_propagation(batch_x)
            self.backward_propagation(self.cost_function.derivative(batch_y, y_pred_train), epoch)
            self.update_epoch(epoch)

    def fit(self, epochs):
        for epoch in range(1, epochs+1):
            batch_train_gen = self.get_batch_generator(self.X_train, self.y_train, self.batch_size)

            self.training_iteration(batch_train_gen, epoch)

            y_pred_train = self.forward_propagation(self.X_train)
            y_pred_test = self.forward_propagation(self.X_test)

            print('loss_train', self.cost_function.compute(self.y_train, y_pred_train), 'acc',
                  self.metric(self.y_train, y_pred_train),
                  "loss_test", self.cost_function.compute(self.y_test, y_pred_test), 'acc',
                  self.metric(self.y_test, y_pred_test))

    def add(self, layer):
        self.layers.append(layer)


def main():
    data_test = pd.read_csv("data/preprocessed/bank/test_bank.csv")
    data_train = pd.read_csv("data/preprocessed/bank/train_bank.csv")
    X_test = data_test.drop('deposit', axis=1).to_numpy()
    y_test = np.expand_dims(data_test['deposit'].to_numpy(), axis=-1)
    X_train = data_train.drop('deposit', axis=1).to_numpy()
    y_train = np.expand_dims(data_train['deposit'].to_numpy(), axis=-1)
    # make_classification(n_samples=6, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, n_classes= 3)
    # enc = OneHotEncoder()
    # y = enc.fit_transform(np.expand_dims(y, axis=-1)).toarray()

    print(X_test.shape)
    print(y_test.shape)

    model = Neural_Net(X_train, y_train, X_test, y_test, accuracy_binary, Binary_Crossentropy(), Adam(0.01,  0.99,0.999,2), 64)

    hidden_layer = Layer(X_test.shape[-1], 8, Sigmoid())
    output_layer = Layer(hidden_layer.output_size, 1, Sigmoid())

    model.add(hidden_layer)
    model.add(output_layer)

    model.fit(500)


if __name__ == '__main__':
    main()
