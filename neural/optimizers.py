import numpy as np

"""
Optimizers, this include the tradicional gradient descent,
and also Momentum, RMS Prop and finally Adam

TODO
-Add Gradient Norm Scaling
-Add Gradient Value Clipping
"""


class Optimizer:
    """
    Abstract base class for all optimizers
    """

    def __init__(self, learning_rate):
        """
        :param learning_rate: Neural network learning rate
        """
        self.learning_rate = learning_rate

    def step(self, *kwargs):
        """
        Function to do one iteration of gradient descent,
        parameters depends on the implementation
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """

    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        :param learning_rate: Neural Network learning rate
        :param momentum: Neural network momentum
        """

        super().__init__(learning_rate)
        self.momentum = momentum
        self.name = "SGD"

    def step(self, gradients, parameters, directions, epoch):
        """
        Function to do one iteration of gradient descent

        :param gradients: Gradients for gradient descent
        :param parameters: Parameters to update
        :param directions: Vectors for the exponentially weighted average
        :param epoch:  Current Epoch, used for correcting bias
        """

        directions["mdW"] = self.momentum * directions["mdW"] + (1 - self.momentum) * gradients["dW"]
        directions["mdB"] = self.momentum * directions["mdB"] + (1 - self.momentum) * gradients["dB"]

        mdw_corrected = directions["mdw"] / (1 - np.power(self.momentum, epoch))
        mdb_corrected = directions["mdB"] / (1 - np.power(self.momentum, epoch))

        parameters["W"] -= self.learning_rate * mdw_corrected
        parameters["B"] -= self.learning_rate * mdb_corrected


class RMSProp(Optimizer):
    """
    RMS Prop Optimizer
    """

    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        :param learning_rate:
        :param beta:
        """
        super().__init__(learning_rate)
        self.beta = beta

    def step(self, gradients, parameters, directions, epoch, epsilon=1e-8):
        """
        Function to do one iteration of gradient descent

        :param gradients: Gradients for gradient descent
        :param parameters: Parameters to update
        :param directions: Vectors for the exponentially weighted average
        :param epoch:  Current Epoch, used for correcting bias
        :param epsilon: Value to prevent division by zero
        """

        directions["rdw"] = self.beta * directions["rdw"] + (1 - self.beta) * np.power(gradients["dw"], 2)
        directions["rdb"] = self.beta * directions["rdb"] + (1 - self.beta) * np.power(gradients["db"], 2)

        rdw_corrected = directions["rdw"] / (1 - np.power(self.beta, epoch))
        rdb_corrected = directions["rdb"] / (1 - np.power(self.beta, epoch))

        delta_W = gradients["db"] / (np.sqrt(rdw_corrected) + epsilon)
        delta_B = gradients["db"] / (np.sqrt(rdb_corrected) + epsilon)

        parameters["W"] -= self.learning_rate * delta_W
        parameters["B"] -= self.learning_rate * delta_B


class Adam(Optimizer):

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        """
        :param learning_rate: Neural network learning rate
        :param beta1: Hyperparameter for adam optimization
        :param beta2: Hyperparameter for adam optimization
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, gradients, parameters, directions, epoch, epsilon=1e-8):
        """
        Function to do one iteration of gradient descent

        :param gradients: Gradients for gradient descent
        :param parameters: Parameters to update
        :param directions: Vectors for the exponentially weighted average
        :param epoch:  Current Epoch, used for correcting bias
        :param epsilon: Value to prevent division by zero
        """

        directions["mdw"] = self.beta1 * directions["mdw"] + (1 - self.beta1) * gradients["dw"]
        directions["mdb"] = self.beta1 * directions["mdb"] + (1 - self.beta1) * gradients["db"]

        directions["rdw"] = self.beta2 * directions["rdw"] + (1 - self.beta2) * np.power(gradients["dw"], 2)
        directions["rdb"] = self.beta2 * directions["rdb"] + (1 - self.beta2) * np.power(gradients["db"], 2)

        mdw_corrected = directions["mdw"] / (1 - np.power(self.beta1, epoch))
        mdb_corrected = directions["mdb"] / (1 - np.power(self.beta1, epoch))

        rdw_corrected = directions["rdw"] / (1 - np.power(self.beta2, epoch))
        rdb_corrected = directions["rdb"] / (1 - np.power(self.beta2, epoch))

        delta_W = mdw_corrected / (np.sqrt(rdw_corrected) + epsilon)
        delta_B = mdb_corrected / (np.sqrt(rdb_corrected) + epsilon)

        parameters["W"] -= self.learning_rate * delta_W
        parameters["B"] -= self.learning_rate * delta_B
