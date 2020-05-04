from neural.activation_functions import *
from neural.optimizers import *

"""
Layer Implementations
"""


class Layer:
    """
    Abstract Base class, will be extended by all Layers
    """

    def __init__(self):
        self.X = None
        self.Y = None
        self.optimizer = None

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for this layer

        :param optimizer: instance of an implementation of abstract class @Optimizer
        """
        assert isinstance(optimizer, Optimizer), "Optimizer has to an instance of ABC Optimizer"
        self.optimizer = optimizer

    def forward_propagation(self, X):
        """
        Propagates through the layer

        :param X: Layer Input
        :return: Layer Output, Y = W.X + B
        """
        raise NotImplementedError

    def back_propagation(self, dY, epoch):
        """
        Backpropagates through the layer

        :param epoch: current epoch
        :param dY: Derivative of the Loss Function with respect to Y
        :return: Derivative of the Loss function with respect to X
        """
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Fully Connected Layer, implementation of ABC Layer
    """

    def __init__(self, input_size, output_size):
        """
        Initializes weights, biases and directions

        :param input_size:
        :param output_size:
        """
        super().__init__()
        self.params = dict()  # Layer parameters
        self.grads = dict()  # grads for gradient descent
        self.directions = dict()  # Vectors for optimizers like RMSProp,Momentum and Adam
        self.input_size = input_size
        self.output_size = output_size

        # # He weight initialization
        self.params["W"] = np.random.randn(self.output_size, self.input_size) * (np.sqrt(2 / self.input_size))
        # bias can be started with zero, doesn't affect much
        self.params["B"] = np.zeros((self.output_size, 1))

        # Initializing self.directions (Even if normal gradient descent is used)
        self.directions["mdW"] = np.zeros(self.params["W"].shape)
        self.directions["mdB"] = np.zeros(self.params["B"].shape)
        self.directions["rdW"] = np.zeros(self.params["W"].shape)
        self.directions["rdB"] = np.zeros(self.params["B"].shape)

    def forward_propagation(self, X):
        self.X = X
        self.Y = np.dot(self.params["W"], self.X) + self.params["B"]
        return self.Y

    def back_propagation(self, dY, epoch):
        self.grads["dX"] = np.dot(self.params["W"].T, dY)
        self.grads["dW"] = np.dot(dY, self.X.T)
        self.grads["dB"] = dY

        self.optimizer.step(self.grads, self.params, self.directions, epoch)

        return self.grads["dX"]


class ActivationLayer(Layer):
    """
    Abstract Base class for all activation layers,
    to extend this class, override the abstract method
    activation
    """

    def __init__(self):
        super().__init__()
        self.gradients = dict()

    def activation(self, X, deriv=False):
        """
        :param X: Input for the activation function
        :param deriv: Boolean to choose if its supposed to return the derivative of the activation function
        :return: The output of the activation function
        """
        pass

    def forward_propagation(self, X):
        self.X = X
        self.Y = self.activation(self.X)
        return self.Y

    def back_propagation(self, dY, epoch):
        self.gradients["dX"] = dY * self.activation(self.X, deriv=True)
        return self.gradients["dX"]


class Relu(ActivationLayer):
    """Wrapper class for activation function Relu"""

    def activation(self, X, deriv=False):
        return relu(X, deriv)


class Sigmoid(ActivationLayer):
    """Wrapper class for activation function Sigmoid"""

    def activation(self, X, deriv=False):
        return sigmoid(X, deriv)


class Softmax(ActivationLayer):
    """Wrapper class for activation function Softmax"""

    def activation(self, X, deriv=False):
        return softmax(X, deriv)
