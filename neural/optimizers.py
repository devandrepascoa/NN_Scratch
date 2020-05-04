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

        mdW_corrected = directions["mdW"] / (1 - np.power(self.momentum, epoch))
        mdB_corrected = directions["mdB"] / (1 - np.power(self.momentum, epoch))

        parameters["W"] -= self.learning_rate * mdW_corrected
        parameters["B"] -= self.learning_rate * mdB_corrected


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

        directions["rdW"] = self.beta * directions["rdW"] + (1 - self.beta) * np.power(gradients["dW"], 2)
        directions["rdB"] = self.beta * directions["rdB"] + (1 - self.beta) * np.power(gradients["dB"], 2)

        rdW_corrected = directions["rdW"] / (1 - np.power(self.beta, epoch))
        rdB_corrected = directions["rdB"] / (1 - np.power(self.beta, epoch))

        delta_W = gradients["dW"] / (np.sqrt(rdW_corrected) + epsilon)
        delta_B = gradients["dB"] / (np.sqrt(rdB_corrected) + epsilon)

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

        directions["mdW"] = self.beta1 * directions["mdW"] + (1 - self.beta1) * gradients["dW"]
        directions["mdB"] = self.beta1 * directions["mdB"] + (1 - self.beta1) * gradients["dB"]

        directions["rdW"] = self.beta2 * directions["rdW"] + (1 - self.beta2) * np.power(gradients["dW"], 2)
        directions["rdB"] = self.beta2 * directions["rdB"] + (1 - self.beta2) * np.power(gradients["dB"], 2)

        mdW_corrected = directions["mdW"] / (1 - np.power(self.beta1, epoch))
        mdB_corrected = directions["mdB"] / (1 - np.power(self.beta1, epoch))

        rdW_corrected = directions["rdW"] / (1 - np.power(self.beta2, epoch))
        rdB_corrected = directions["rdB"] / (1 - np.power(self.beta2, epoch))

        delta_W = mdW_corrected / (np.sqrt(rdW_corrected) + epsilon)
        delta_B = mdB_corrected / (np.sqrt(rdB_corrected) + epsilon)

        parameters["W"] -= self.learning_rate * delta_W
        parameters["B"] -= self.learning_rate * delta_B
