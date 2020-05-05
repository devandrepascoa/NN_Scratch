import numpy as np

"""
Loss functions
"""


def cross_entropy(YHat, Y, deriv=False):
    if deriv:
        return - (np.divide(Y, YHat) - np.divide(1 - Y, 1 - YHat))
    else:
        M = Y.shape[1]
        # Log probabilities
        loss_sum = np.sum(np.multiply(Y, np.log(YHat)))
        cost = -(1 / M) * loss_sum
        return cost


def mse(YHat, Y, deriv=False):
    if deriv:
        return 2 * (YHat - Y) / Y.size
    else:
        return np.mean(np.power(Y - YHat, 2))
