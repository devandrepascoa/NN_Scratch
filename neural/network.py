import pickle

import neural
import numpy as np

from neural import math_utils, losses, Optimizer, datasets
from neural.Layers.layers import Dropout, Layer


class Network:
    """
    Neural network class, used for creating models
    """

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.input_size = None

    def save(self, path="neural_network"):
        params = []
        for layer in self.layers:
            if layer.is_weighted:
                params.append(layer.params)
        params = np.array(params)
        with open(path, 'wb') as fout:
            save_dic = {"params": params, "version": neural.__version__}
            pickle.dump(save_dic, fout)
            fout.close()

    def load(self, path="neural_network"):
        with open(path, 'rb') as fin:
            saved_dic = pickle.load(fin)
            # Validations
            assert neural.__version__ == saved_dic[
                "version"], "Incompatible version\nCurrent version: " + neural.__version__ + \
                            "\nLoaded version: " + saved_dic["version"]
            params = saved_dic["params"]
            counter = 0
            for layer in self.layers:
                if layer.is_weighted:
                    layer.params = params[counter]
                    counter += 1
            fin.close()

    def forward_propagation(self, X, train_mode=True):
        """
        Forward propagation implementation

        :param train_mode: boolean to disable dropout
        :param X: Input
        :return: Neural network Prediction(Yhat)
        """
        M = X.shape[-1]
        output = X
        for layer in self.layers:
            if not train_mode and isinstance(layer, Dropout):
                continue
            output = layer.forward_propagation(output, layer.params, M)

        return output

    def back_propagation(self, YHat, Y, epoch):
        """
        Back propagation through all layers

        :param YHat: Neural network prediction
        :param Y: Ground Truth
        :param epoch: current epoch
        :return: Input Gradients
        """
        M = Y.shape[-1]
        grad = self.loss(YHat, Y, deriv=True)
        for layer in reversed(self.layers):
            grad = layer.back_propagation(layer.X, layer.params, grad, M, epoch)
        return grad

    def add(self, layer):
        """
        Function to append a layer to the model

        :param layer: Instance of ABC Layer
        """
        assert isinstance(layer, Layer), "Layer is not an instance of ABC Layer"
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        """
        Sets optimizer and loss function

        :param optimizer: optimizer, must be instance of ABC Optimizer
        :param loss: loss function, check losses.py
        """
        self.optimizer = optimizer
        self.loss = loss

        assert isinstance(optimizer, Optimizer), "Selected optimizer must be instance of ABC Optimizer"

        for layer in self.layers:
            layer.set_optimizer(optimizer)

    def evaluate(self, dataset):
        """
        Function to evaluate the neural network on a certain dataset

        :param dataset:
        :return: dictionary{
         cost -> forward propagation cost,
         accuracy -> % of data points the NN got right
        }
        """
        (X, Y) = dataset
        output = self.forward_propagation(X)
        cost = losses.cross_entropy(output, Y)
        accuracy = math_utils.get_accuracy(output, Y)
        return {"cost": cost, "accuracy": accuracy}

    def predict(self, input_data):
        """
        (data,number_of_training_examples)
        Function to predict for a single input

        :param input_data:
        :return:
        """
        assert input_data.shape == (self.input_size, 1)
        X = input_data
        output = self.forward_propagation(X)
        return output

    def fit(self, dataset, epochs=500, batch_size=1, val_dataset=None, print_costs=True):
        """
        Function to train the neural network to fit the training dataset

        :param print_costs: Boolean to print data
        :param dataset: input dataset, has to be of shape (data,training_examples)
        :param epochs: Number of epochs
        :param batch_size: Mini batch size, number of training examples before optimization
        :param val_dataset: Test dataset, will print validation data, has to be of shape as dataset except
        number of examples
        """

        assert self.optimizer is not None, "Model not compiled"

        # Dataset validation
        val_enabled = True if val_dataset is not None else False
        x_train, y_train = dataset
        x_test, y_test = None, None
        if val_enabled:
            x_test, y_test = val_dataset

        X_test, Y_test = None, None
        assert len(x_train.shape) == 2
        assert len(y_train.shape) == 2
        if val_enabled:
            (X_test, Y_test) = val_dataset
            assert len(X_test.shape) == 2, "Incorrect data shape"
            assert len(Y_test.shape) == 2, "Incorrect data shape"

        datasets.validate_dataset(dataset, "Data set")
        if val_enabled:
            datasets.validate_dataset(val_dataset, "Validation set")

        self.input_size = self.layers[0].input_size
        # Epoch iteration
        for i in range(1, epochs):
            cost = 0
            accuracy = 0
            counter = 0
            for X, Y in math_utils.calculate_batches(x_train, y_train, batch_size):
                # ForwardPropagation
                YHat = self.forward_propagation(X)

                cost += self.loss(YHat, Y)
                accuracy += math_utils.get_accuracy(YHat, Y)

                # Backpropagation
                self.back_propagation(YHat, Y, i)

                counter += 1
            cost /= counter
            accuracy /= counter

            # Printing Info
            if print_costs:
                # Printing average accuracy for mini batch and current accuracy for batched gradient descent
                print("Epoch:{},Cost:{}, Accuracy:{}".format(i, cost, accuracy))
                if val_enabled:  # Validation testing
                    YHat_test = self.forward_propagation(x_test)
                    err_test = self.loss(YHat_test, y_test)
                    accuracy_test = math_utils.get_accuracy(YHat_test, y_test)
                    print("Validation Cost:{}, Validation Accuracy:{}".format(err_test,
                                                                              accuracy_test))
