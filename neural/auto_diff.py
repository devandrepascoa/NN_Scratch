import numpy as np


class MultiplyGate:
    def forward_propagation(self, X, W):
        return np.dot(X, W)

    def backward_propagation(self, X, W, dZ):
        pass
