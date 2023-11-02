# THIS FILE IS A MESS
import numpy as np

alpha = 6

def gradient_descent(x, dx, epoch):
    return x - np.clip(0.99 ** epoch * alpha * dx, -1, 1)