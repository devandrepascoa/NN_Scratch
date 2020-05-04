import numpy as np

"""
Activation Functions, including their respective derivative
"""


def sigmoid(X, deriv=False):
    if deriv:
        return sigmoid(X) * (1 - sigmoid(X))
    else:
        return 1 / (1 + np.exp(-X))


def relu(X, deriv=False):
    if deriv:
        X[X < 0] = 0
        return X
    else:
        X[X <= 0] = 0
        X[X > 0] = 1
        return X


def softmax(X, deriv=False):
    if deriv:
        return softmax(X) * (1 - softmax(X))
    else:
        e_X = np.exp(X - np.max(X))
        return e_X / np.sum(e_X, axis=0)
