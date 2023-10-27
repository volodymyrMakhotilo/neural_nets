# THIS FILE IS A MESS
import  numpy as np

alpha = 6

def gradient_descent(x, dx, epoch):
    return x - 0.99 ** np.sqrt(epoch) * alpha * dx