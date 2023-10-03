import numpy as np
from numpy.random import uniform, randn
from utils.cost_functions import Sparse_Categorical_Crossentropy
from utils.optimizers import gradient_descent
from utils.activations import ReLU, Softmax
from tqdm import tqdm


class Layer:
    # Shapes should be compatible for matrix multiplication
    def __init__(self, input_size, output_size, activation):
        self.batch_size = 0
        self.output_size = output_size
        self.bias = self.weights_init(1, output_size)
        self.weights = self.weights_init(input_size, output_size)
        self.activation = activation
        self.cached_input = np.NaN

    def forward(self, X):
        self.batch_size = X.shape[0]
        self.cached_input = X
        return self.compute_linear(X)

    def compute_linear(self, X):
        return self.activation.compute(np.matmul(X, self.weights) + self.bias)

    def update(self, grad):
        dW = np.matmul(self.cached_input.T, grad) / self.batch_size
        db = np.mean(grad, axis=0)
        self.weights = gradient_descent(self.weights, dW)
        self.bias = gradient_descent(self.bias, db)

    def backward(self, dX):

      grad = dX * self.activation.derivative(self.compute_linear(self.cached_input))

      self.update(grad)
      return np.matmul(grad, self.weights.T)

    # Experiment!
    def weights_init(self, height, width):
        return uniform(0.1, 1., (height, width))

class Neural_Net:
    # FIX COST AND OPT INIT
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.layers = []
        #SPECIFY COST
        self.cost_function = Sparse_Categorical_Crossentropy()
        #SPECIFY OPTIMIZER
        self.optimizer = gradient_descent

    # Didn't put much thought in code
    # Batch is not coded
    def forward_propagation(self, batch):
        output = batch
        for layer in self.layers:
            output = layer.forward(output)
        print(self.cost_function.compute(self.y_train, output))
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

    #Batch
    def fit(self, epochs):
        for _ in range(epochs):
            y_pred = self.forward_propagation(self.X_train)
            self.backward_propagation(self.cost_function.derivative(self.y_train, y_pred))

    def add(self, layer):
        self.layers.append(layer)

def main():
    X = randn(300, 2)
    y = (randn(300, 3) > 0.5).astype(np.uint)
    model = Neural_Net(X, y)

    input_layer = Layer(X.shape[-1], 3, ReLU())
    hidden_layer = Layer(input_layer.output_size, 5, ReLU())
    output_layer = Layer(hidden_layer.output_size, 3, Softmax())

    model.add(input_layer)
    model.add(hidden_layer)
    model.add(output_layer)


    model.fit(10)


if __name__ == '__main__':
    main()