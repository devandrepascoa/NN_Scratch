"""
Layer Implementations
"""
from neural.activation_functions import *
from neural.optimizers import Optimizer
import numpy as np


class Layer:
    """
    Abstract Base class, will be extended by all Layers
    """

    def __init__(self):
        self.X = None
        self.Y = None
        self.params = None
        self.grads = None
        self.is_weighted = False
        self.optimizer = None

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for this layer

        :param optimizer: instance of an implementation of abstract class @Optimizer
        """
        assert isinstance(optimizer, Optimizer), "Optimizer has to an instance of ABC Optimizer"
        self.optimizer = optimizer

    def forward_propagation(self, X, params, M):
        """
        Propagates through the layer

        :param M: Neural network batch size
        :param params: Layer's weights and biases
        :param X: Layer Input
        :return: Layer Output, Y = W.X + B
        """
        raise NotImplementedError

    def back_propagation(self, X, params, dY, M, epoch):
        """
        Backpropagates through the layer

        :param M: Neural network batch size
        :param X: Input data for this layer
        :param params: Layer weights and biases
        :param epoch: current epoch
        :param dY: Derivative of the Loss Function with respect to the current layer ouput
        :return: Derivative of the Loss function with respect to the current layer input
        """
        raise NotImplementedError


class Dense(Layer):
    """
    Fully Connected Layer, implementation of ABC Layer
    """

    def __init__(self, input_size, output_size, weights_init="Xavier"):
        """
        Initializes weights, biases and params

        :param weights_init: String to choose the desired weights initializer,
            He-> He weight initialization(Good for Relu activation function)
            Xavier -> Xavier weight initialization(Good for Tanh activation function)
            Random -> Random weight initialization
        :param input_size: This layer's input size
        :param output_size: This layer's output size
        """
        super().__init__()
        self.is_weighted = True
        self.params = dict()  # Layer parameters
        self.grads = dict()  # grads for gradient descent
        self.input_size = input_size
        self.output_size = output_size

        # # He weight initialization
        if weights_init == "He":
            self.params["W"] = np.random.randn(self.output_size, self.input_size) * (np.sqrt(2 / self.input_size))
        elif weights_init == "Xavier":
            self.params["W"] = np.random.randn(self.output_size, self.input_size) * (
                np.sqrt(1 / (self.input_size + self.output_size)))
        elif weights_init == "Random":
            self.params["W"] = np.random.randn(self.output_size, self.input_size)
        # bias can be started with zero, doesn't affect much
        self.params["B"] = np.zeros((self.output_size, 1))

        # Initializing vectors for Adam, Rms etc (Even if normal gradient descent is used)
        self.grads["mdW"] = np.zeros(self.params["W"].shape)
        self.grads["mdB"] = np.zeros(self.params["B"].shape)
        self.grads["rdW"] = np.zeros(self.params["W"].shape)
        self.grads["rdB"] = np.zeros(self.params["B"].shape)

    def forward_propagation(self, X, params, M):
        self.X = X
        self.Y = np.dot(params["W"], X) + params["B"]

        return self.Y

    def back_propagation(self, X, params, dY, M, epoch):  # dY = dz
        self.grads["dX"] = np.dot(params["W"].T, dY)
        self.grads["dW"] = np.dot(dY, X.T) / M
        self.grads["dB"] = np.sum(dY, axis=1, keepdims=True) / M

        self.optimizer.step(self.grads, params, epoch)

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

    def forward_propagation(self, X, params, M):
        self.X = X
        self.Y = self.activation(self.X, deriv=False)
        return self.Y

    def back_propagation(self, X, params, dY, M, epoch):
        self.gradients["dX"] = dY * self.activation(X, deriv=True)

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


class Tanh(ActivationLayer):
    """Wrapper class for activation function Softmax"""

    def activation(self, X, deriv=False):
        return tanh(X, deriv)


class Dropout(Layer):
    """Dropout layer, deactivates multiple neurons based on a probability"""

    def __init__(self, keep_prob=0.5):
        super().__init__()
        self.keep_prob = keep_prob
        self.D = None

    def forward_propagation(self, X, params, M):
        self.D = np.random.rand(X.shape[0],
                                X.shape[1]) < self.keep_prob
        self.X = X
        self.Y = (self.X * self.D) / self.keep_prob
        return self.Y

    def back_propagation(self, X, params, dY, M, epoch):
        dX = dY * self.D
        dX /= self.keep_prob
        return dX


class Flatten(Layer):
    """
    Flattens Input data, typically from a Convolutional model
    into a fully connected one
    """

    def forward_propagation(self, X, params, M):
        self.X = X
        self.Y = X.reshape((-1, M))
        return self.Y

    def back_propagation(self, X, params, dY, M, epoch):
        return dY.reshape(X.shape)


class Conv2D(Layer):
    """
    Does a entire convolution on the input
    """

    def __init__(self, input_shape, filter_shape, num_filters):
        super().__init__()
        self.input_shape = input_shape
        self.input_depth = input_shape.shape[-1]
        self.filter_shape = filter_shape
        self.num_filters = num_filters

    def forward_propagation(self, X, params, M):
        self.X = X

    def back_propagation(self, X, params, dY, M, epoch):
        pass