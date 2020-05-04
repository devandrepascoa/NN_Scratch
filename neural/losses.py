import numpy as np

"""
Loss functions
"""


def cross_entropy(YHat, Y, deriv=False):
    if deriv:
        return -(Y / YHat) + (1 + Y) / (1 + YHat)
    else:
        M = YHat.shape[1]
        logprobs = np.multiply(np.log(YHat), Y)
        cost = - np.sum(logprobs) / M
        return float(np.squeeze(cost))


def mse(YHat, Y, deriv=False):
    if deriv:
        return 2 * (YHat - Y) / Y.size
    else:
        return np.mean(np.power(Y - YHat, 2))
