import pickle
import sys

import numpy as np
import tensorflow.keras as keras

__version__ = '1.0.0'

from neural.initializations import *
from neural.optimizers import *


def loadMnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], -1)).T / 255.0
    x_test = (x_test.reshape(x_test.shape[0], -1)).T / 255.0
    y_train = (y_train.reshape(y_train.shape[0], 1)).T
    y_test = (y_test.reshape(y_test.shape[0], 1)).T
    y_train = MathUtils.hotOne(y_train, 10)
    y_test = MathUtils.hotOne(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def loadCifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = (x_train.reshape(x_train.shape[0], -1) / 255.0).T
    x_test = (x_test.reshape(x_test.shape[0], -1) / 255.0).T
    y_train = (y_train.reshape(y_train.shape[0], 1)).T
    y_test = (y_test.reshape(y_test.shape[0], 1)).T
    y_train = MathUtils.hotOne(y_train, 10)
    y_test = MathUtils.hotOne(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


class MathUtils():

    @staticmethod
    def hotOne(array, output_size):  # Creates a hot One array from single digit
        assert len(array.shape) == 2, "Input has to have shape (data,data_size)"
        Y_orig = array
        Y = np.zeros((output_size, Y_orig.shape[-1]))
        for i in range(0, Y_orig.shape[1]):
            value = Y_orig[0, i]
            Y[value, i] = 1.0
        return Y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return MathUtils.sigmoid(x) * (1 - MathUtils.sigmoid(x))

    @staticmethod
    def relu(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def relu_deriv(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    @staticmethod
    def softmax_deriv(x):
        return MathUtils.softmax(x) * (1 - MathUtils.softmax(x))

    @staticmethod
    def cross_entropy(A, Y):
        M = A.shape[1]
        logprobs = np.multiply(np.log(A), Y)
        cost = - np.sum(logprobs) / M
        return float(np.squeeze(cost))

    @staticmethod
    def cross_entropy_deriv(A, Y):
        return -(Y / A) + (1 + Y) / (1 + A)

    @staticmethod
    def l2_regularization(params, lambd, M):
        sum = 0
        for key in params:
            if key.startswith("W"):
                sum += np.sum(np.square(params[key]))
        return lambd * sum / (2 * M)


# TODO
# Improve oop architecture !! high priority
# Fix gradients(0.3 dif)
# Work on other activation functions and optimizers
# Apply Mini batches

class NN(object):

    # Function to do a full forward propagation with the current parameters
    def forward_propagate(self, X, params, train_mode, Y=None, lambd=0.7):
        cache = dict()
        length = len(self.S)  # neural network length
        M = X.shape[1]  # Number of training examples

        # First Layer
        cache["Z1"] = np.dot(params["W1"], X) + params["B1"]
        cache["A1"] = MathUtils.relu(cache["Z1"])
        if train_mode and self.enable_dropout:  # Dropping 50% of neurons if in training mode and dropout mode
            self.dropout_fwd(cache, "1", self.dropout_value)

        for i in range(2, length - 1):
            cache["Z" + str(i)] = np.dot(params["W" + str(i)],
                                         cache["A" + str(i - 1)]) + params["B" + str(i)]
            cache["A" + str(i)] = MathUtils.relu(cache["Z" + str(i)])
            if train_mode and self.enable_dropout:
                self.dropout_fwd(cache, str(i), self.dropout_value)

        # Output layer
        cache["Z" + str(length - 1)] = np.dot(params["W" + str(length - 1)],
                                              cache["A" + str(length - 2)]) + params["B" + str(length - 1)]
        cache["A" + str(length - 1)] = MathUtils.softmax(cache["Z" + str(length - 1)])
        Yhat = cache["A" + str(length - 1)]  # Final prediction

        if Y is not None:
            # Computing costs (L2 regularization included)
            cache["cost"] = MathUtils.cross_entropy(Yhat, Y) + MathUtils.l2_regularization(params, lambd, M)
            cache["accuracy"] = self.get_accuracy(Yhat, Y)
        return cache

    # Back propagation using Gradient descent
    def back_propagate(self, X, Y, cache, parameters, lambd=0.7):
        gradients = dict()
        length = len(self.S)  # neural network length
        M = X.shape[1]  # Number of training examples

        # Gradients for activations and before applying activations
        Yhat = cache["A" + str(length - 1)]  # Predicted Output
        gradients["dz" + str(length - 1)] = (Yhat - Y)
        for i in range(2, length):
            gradients["da" + str(length - i)] = np.dot(parameters["W" + str(length - i + 1)].T,
                                                       gradients["dz" + str(length - i + 1)])
            if self.enable_dropout:
                self.dropout_bw(cache, gradients, str(length - i),
                                self.dropout_value)  # Dropping out 50% of the neurons
            gradients["dz" + str(length - i)] = gradients["da" + str(length - i)] * MathUtils.relu_deriv(
                cache["Z" + str(length - i)])

        # Gradients for weights and biases
        # dot devido a ser a soma remember my dude produto escalar
        gradients["dw1"] = (1.0 / M) * np.dot(gradients["dz1"], X.T) + (lambd * parameters["W1"]) / M
        gradients["db1"] = (1 / M) * np.sum(gradients["dz1"], axis=1, keepdims=True)
        for i in range(2, length):
            gradients["dw" + str(i)] = (1 / M) * np.dot(gradients["dz" + str(i)], cache["A" + str(i - 1)].T)
            gradients["db" + str(i)] = (1 / M) * np.sum(gradients["dz" + str(i)], axis=1, keepdims=True)

        return gradients

    # Function to calculate accuracy based on the model predictions and real output
    def get_accuracy(self, Yhat, Y):
        m = Yhat.shape[1]
        sum = 0
        Yhat_max = np.argmax(Yhat, axis=0)
        Y_max = np.argmax(Y, axis=0)
        for i in range(0, m):
            if Yhat_max[i] == Y_max[i]:
                sum += 1
        return (sum / m) * 100.0

    # Function to evaluate the neural network on a certain dataset
    def evaluate(self, dataset):
        (X, Y) = dataset
        cache = self.forward_propagate(X, self.params, False, Y)
        cost = MathUtils.cross_entropy(cache["A" + str(len(self.S) - 1)], Y)
        accuracy = self.get_accuracy(cache["A" + str(len(self.S) - 1)], Y)
        return {"cache": cache, "cost": cost, "accuracy": accuracy}

    # (784, 1) <--input data shape in case of mnist
    # Function to predict for a single input
    def predict(self, input_data):
        assert input_data.shape == (self.S[0], 1)
        X = input_data
        cache = self.forward_propagate(X, self.params, False)
        label = cache["A" + str(len(self.S) - 1)]
        return label

    @staticmethod
    def validate_dataset(dataset, name):
        assert isinstance(dataset, tuple) and len(dataset) == 2, (
                name + " has to be a tuple of size 2 -> (x_train,y_train)")
        assert isinstance(dataset[0], np.ndarray) and isinstance(dataset[1], np.ndarray), (
                name + " data has to be numpy array")
        assert len(dataset[0].shape) == 2 and len(dataset[1].shape) == 2, (
                name + " data has to be of shape (data_size,data) and labels (label_size,label)")

    # Function to save the neural network object(Will add only weights and biases in future)
    def save(self, path="neural_network"):
        with open(path, 'wb') as fout:
            save_dic = {"params": self.params, "size": self.S, "version": __version__}
            pickle.dump(save_dic, fout)
            fout.close()

    # Function to load a neural network object
    def load(self, path="neural_network"):
        with open(path, 'rb') as fin:
            saved_dic = pickle.load(fin)
            # Validations
            assert __version__ == saved_dic["version"], "Incompatible version\nCurrent version: " + __version__ + \
                                                        "\nLoaded version: " + saved_dic["version"]
            assert self.S == saved_dic["size"], "Network model does not match,\nLoaded model shape:" \
                                                + str(saved_dic["size"]) + "\nModel shape:" + str(self.S)
            self.params = saved_dic["params"]
            fin.close()

    def calculate_batches(self, X, Y, mini_batch_size, shuffle=False):
        M = X.shape[1]  # Number of training examples
        if mini_batch_size == M:
            yield X, Y, 1
            return
        counter = 0  # Counter
        # Shuffles the dataset with synchronization (X,Y)
        if shuffle:
            permutation = list(np.random.permutation(M))
            X = X[:, permutation]
            Y = Y[:, permutation].reshape((Y.shape[0], M))
        # Partitions the dataset into batches of size mini_batch_size
        num_complete_batches = M // mini_batch_size
        # Yield all complete batches(batches of exact size = mini_batch_size
        for k in range(0, num_complete_batches):
            mini_batch_X = X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            counter += 1
            yield mini_batch_X, mini_batch_Y, counter
        # In case there is one batch that is not complete yield it
        if M % mini_batch_size != 0:
            mini_batch_X = X[:, num_complete_batches * mini_batch_size:]
            mini_batch_Y = Y[:, num_complete_batches * mini_batch_size:]
            counter += 1
            yield mini_batch_X, mini_batch_Y, counter
        return X, Y

    # function to apply gradient checking algorithm to assert that the gradient are correct
    def gradient_check(self, parameters, gradients, X, Y, epsilon=1e-7):
        grad_approx = []
        grad = np.array([])
        count = 0
        # Preparing necessary analytic gradient values (only require dW and dB)
        for i in gradients.keys():  # Adding every weight and bias (activations and outputs excluded)
            if i.startswith("dw") or i.startswith("db"):
                new_vector = np.reshape(gradients[i], (-1, 1))
                if count == 0:
                    grad = new_vector
                else:
                    grad = np.concatenate((grad, new_vector), axis=0)
                count = count + 1
        grad = np.array(grad)  # Array of gradients to compare to approximated gradients

        # Building the numerical gradient approximations
        for i in parameters.keys():
            for idx in np.ndindex(parameters[i].shape):
                thetaplus = parameters[i][idx] + epsilon  # calculating theta plus for each parameter
                modified_params = parameters.copy()
                modified_params[i][idx] = thetaplus
                # testing network based on modified params
                cache = self.forward_propagate(X, modified_params, True, Y)
                J_Plus = cache["cost"]

                thetaminus = parameters[i][idx] - epsilon
                modified_params = parameters.copy()
                modified_params[i][idx] = thetaminus
                cache = self.forward_propagate(X, modified_params, True, Y)
                J_Minus = cache["cost"]
                # Adding the approximation to a list
                grad_approx.append((J_Plus - J_Minus) / (2 * epsilon))

        grad_approx = np.array(grad_approx).reshape(-1, 1)
        # Comparing values for debugging
        # for i in range(0, grad.shape[0]):
        #    print("Value: {}, Real value: {}".format(grad[i], grad_approx[i]))

        # Calculating relative error
        numerator = np.linalg.norm(grad - grad_approx)  # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)  # Step 2'
        difference = numerator / denominator  # Step 3'

        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

    # Inverse dropout for forward propagation
    @staticmethod
    def dropout_fwd(cache, activation_layer, keep_prob):
        cache["D" + activation_layer] = np.random.rand(cache["A" + activation_layer].shape[0],
                                                       cache["A" + activation_layer].shape[1]) < keep_prob
        cache["A" + activation_layer] *= cache["D" + activation_layer]
        cache["A" + activation_layer] /= keep_prob

    # Inverse dropout for forward propagation
    @staticmethod
    def dropout_bw(cache, gradients, activation_layer, keep_prob):
        gradients["da" + activation_layer] *= cache["D" + activation_layer]
        gradients["da" + activation_layer] /= keep_prob

    def __init__(self, shape):
        if shape is not None:
            self.S = shape
            assert len(self.S) > 2

    def fit_iteration(self, epochs, X, Y, X_test, Y_test):
        for i in range(1, epochs):
            cache = None
            avg_accuracy = 0  # average accuracy for mini batches
            counter = 0  # Counter for averaging accuracy
            for x, y, z in self.calculate_batches(X, Y, self.mini_batch_size):  # For every mini batch
                cache = self.forward_propagate(x, self.params, True, y)
                grads = self.back_propagate(x, y, cache, self.params)
                if self.check_grads:
                    self.gradient_check(self.params, grads, x, y)

                # Initializing directions
                if self.optimizer == "adam":
                    Adam_Optimizer(self, i, grads, self.params, self.learning_rate)
                elif self.optimizer == "rms":
                    RMS_Prop(self, i, grads, self.params, self.learning_rate)
                elif self.optimizer == "momentum":
                    gradient_descent_with_momentum(self, i, grads, self.params, self.learning_rate)
                elif self.optimizer == "gd":
                    gradient_descent(self, grads, self.params, self.learning_rate)
                else:
                    raise AssertionError("Wrong Optimizer selected " + self.optimizer)

                avg_accuracy += cache["accuracy"]
                counter = z
            avg_accuracy /= counter

            if self.print_costs:
                # Printing average accuracy for mini batch and current accuracy for batched gradient descent
                print("Epoch:{},Cost:{}, Accuracy:{}".format(i, cache["cost"], avg_accuracy))
                if self.validate_enabled:
                    cache_test = self.forward_propagate(X_test, self.params, False, Y_test)
                    print("Validation Cost:{}, Validation Accuracy:{}".format(cache_test["cost"],
                                                                              cache_test["accuracy"]))

    # (data, number_of_samples) <--input data shape ex:(784,60000) for mnist
    # [input_size,hidden_layers,output_size] <--shape list shape ex:[784,128,128,10] for mnist
    # In the case of using Momentum or RMSProp, Beta1 will be the related hyperparameter
    # In the case of using Adam optimizer, both Beta1 and Beta2 will be used
    def fit(self, dataset, optimizer="adam", val_dataset=None, dropout_value=0.8, check_grads=False,
            early_stopping=True, Beta1=0.9, Beta2=0.999,
            print_costs=True, epochs=2000, learning_rate=0.01, mini_batch_active=False,
            mini_batch_size=64, enable_dropout=True, save_enabled=True):
        self.enable_dropout = enable_dropout
        self.dropout_value = dropout_value
        self.check_grads = check_grads
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.print_costs = print_costs
        self.save_enabled = save_enabled
        self.validate_enabled = True if val_dataset is not None else False
        self.mini_batch_active = mini_batch_active
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.Beta1 = Beta1
        self.Beta2 = Beta2

        # Validations
        NN.validate_dataset(dataset, "Data set")
        if self.validate_enabled:
            NN.validate_dataset(val_dataset, "Validation set")

        assert not (check_grads and enable_dropout), (
            "Gradient checking and Dropout regularization cannot be both enabled at the same time.")

        (X, Y) = dataset
        X_test, Y_test = None, None
        assert len(X.shape) == 2
        assert len(Y.shape) == 2
        if self.validate_enabled:
            (X_test, Y_test) = val_dataset
            assert len(X_test.shape) == 2, "Incorrect data shape"
            assert len(Y_test.shape) == 2, "Incorrect data shape"

        assert self.S[0] == X.shape[0] and self.S[-1] == Y.shape[0], "Incorrect data shape"

        # Initializing parameters
        self.params = init_params(self)

        # Initializing directions
        if optimizer == "adam":
            self.directions = init_params_adam(self)
        elif optimizer == "rms":
            self.directions = init_params_rms(self)
        elif optimizer == "momentum":
            self.directions = init_params_momentum(self)
        elif optimizer == "gd":
            pass
        else:
            raise AssertionError("Wrong Optimizer selected " + optimizer)

        if not self.mini_batch_active:
            self.mini_batch_size = X.shape[1]

        print("Starting training, NN dimensions: {}".format(self.S))

        # Training the neural network, in case of keyboard interrupt save the neural network
        try:
            self.fit_iteration(epochs, X, Y, X_test, Y_test)
        except KeyboardInterrupt:
            print("Interrupted, saving parameters")
            if self.save_enabled:
                self.save()
            sys.exit()
