import numpy as np


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def logistic(x):
    return 1 / (1 + np.exp(-x))
