from neural import math_utils
from neural.layers import Layer


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def forward_propagation(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def back_propagation(self, YHat, Y, epoch):
        grad = self.loss(YHat, Y, deriv=True)
        for layer in reversed(self.layers):
            grad = layer.back_propagation(grad, epoch)
        return grad

    def add(self, layer):
        """
        Function to append a layer to the model

        :param layer: Instance of ABC Layer
        """
        assert isinstance(layer, Layer), "Layer is not an instance of ABC Layer"
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

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
        val_enabled = True if val_dataset is not None else False

        x_train, y_train = dataset
        x_test, y_test = None, None
        if val_enabled:
            x_test, y_test = val_dataset

        for i in range(1, epochs):
            cost = 0
            accuracy = 0
            for X, Y in math_utils.calculate_batches(x_train, y_train, batch_size):
                # ForwardPropagation
                YHat = self.forward_propagation(X)

                cost += self.loss(YHat, Y)
                accuracy += math_utils.get_accuracy(YHat, Y)

                # Backpropagation
                grad = self.back_propagation(YHat, Y, i)

            cost /= batch_size
            accuracy /= batch_size
            if print_costs:
                # Printing average accuracy for mini batch and current accuracy for batched gradient descent
                print("Epoch:{},Cost:{}, Accuracy:{}".format(i, cost, accuracy))
                if val_enabled:
                    YHat_test = self.forward_propagation(x_test)
                    err_test = self.loss(YHat_test, y_test)
                    accuracy_test = math_utils.get_accuracy(YHat_test, y_test)
                    print("Validation Cost:{}, Validation Accuracy:{}".format(err_test,
                                                                              accuracy_test))
