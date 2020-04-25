import pickle
import numpy as np
import tensorflow.keras as keras


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
    def flatten_dic(dic):
        keys = []
        count = 0
        theta = np.array([])
        for i in dic.keys():
            new_vector = np.reshape(dic[i], (-1, 1))
            keys = keys + [i] * new_vector.shape[0]
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
        return theta, keys

    @staticmethod
    def vector_to_dic(vector, keys):
        return None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return MathUtils.sigmoid(x) * (1 - MathUtils.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        # return np.greater(x, 0).astype(int)
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def softmax(x):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=0, keepdims=True)
        return x_exp / x_sum

    @staticmethod
    def hotOne(array, output_size):
        assert len(array.shape) == 2, "Input has to have shape (data,data_size)"
        Y_orig = array
        Y = np.zeros((output_size, Y_orig.shape[-1]))
        for i in range(0, Y_orig.shape[1]):
            value = Y_orig[0, i]
            Y[value, i] = 0.999999999
        return Y

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

    @staticmethod  # normalize para dados do tipo (data,number_samples)
    def normalize(data):
        data = data - MathUtils.media(data)
        constant = MathUtils.media((data ** 2))
        return data / constant

    @staticmethod  # media para dados do tipo (data,number_samples)
    def media(data):
        soma = np.sum(data, axis=0)
        return soma / data.shape[0]


# TODO
# Improve oop architecture !! high priority
# Add regularization in the form of L2 regularization and inverse dropout
class NN(object):
    # Function to initialize our parameters( Weights and biases)
    def init_params(self, S):
        parameters = {}
        for i in range(1, len(S)):
            # (np.sqrt(2 / S[i - 1])) important for weight initialization, using He initialization(NOT Xavier)
            parameters["W" + str(i)] = np.random.randn(S[i], S[i - 1]) * (np.sqrt(2 / S[i - 1]))
            parameters["B" + str(i)] = np.zeros((S[i], 1))  # bias can be started with zero, doesn't affect much
        return parameters

    # Function to do a full forward propagation with the current parameters
    def forward_propagate(self, X, params, S, train_mode):
        cache = {}
        # First layer(not the input)
        cache["Z1"] = np.dot(params["W1"], X) + params["B1"]
        cache["A1"] = MathUtils.relu(cache["Z1"])
        if train_mode and self.enable_dropout:
            self.dropout_fwd(cache, "1", 0.5)

        for i in range(2, len(S) - 1):
            cache["Z" + str(i)] = np.dot(params["W" + str(i)],
                                         cache["A" + str(i - 1)]) + params["B" + str(i)]
            cache["A" + str(i)] = MathUtils.relu(cache["Z" + str(i)])
            if train_mode and self.enable_dropout:
                self.dropout_fwd(cache, str(i), 0.5)

        # Output layer
        cache["Z" + str(len(S) - 1)] = np.dot(params["W" + str(len(S) - 1)],
                                              cache["A" + str(len(S) - 2)]) + params["B" + str(len(S) - 1)]
        cache["A" + str(len(S) - 1)] = MathUtils.softmax(cache["Z" + str(len(S) - 1)])
        return cache

    # Back propagation using Gradient descent
    def back_propagate(self, X, Y, cache, parameters, S):
        gradients = {}
        M = Y.shape[1]
        gradients["dz" + str(len(S) - 1)] = (cache["A" + str(len(S) - 1)] - Y) / M
        for i in range(2, len(S)):
            gradients["da" + str(len(S) - i)] = np.dot(parameters["W" + str(len(S) - i + 1)].T,
                                                       gradients["dz" + str(len(S) - i + 1)])
            if self.enable_dropout:
                self.dropout_bw(cache, gradients, str(len(S) - i), 0.5)
            gradients["dz" + str(len(S) - i)] = gradients["da" + str(len(S) - i)] * MathUtils.relu_deriv(
                cache["Z" + str(len(S) - i)])

        gradients["dw1"] = np.dot(gradients["dz1"], X.T)  # dot devido a ser a soma remember my dude produto escalar
        gradients["db1"] = np.sum(gradients["dz1"], axis=1, keepdims=True)
        for i in range(2, len(S)):
            gradients["dw" + str(i)] = np.dot(gradients["dz" + str(i)], cache["A" + str(i - 1)].T)
            gradients["db" + str(i)] = np.sum(gradients["dz" + str(i)], axis=1, keepdims=True)

        return gradients

    # Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
    def learn(self, gradients, parameters, learning_rate, network_dims):
        for i in range(1, len(network_dims)):
            parameters["W" + str(i)] -= learning_rate * gradients["dw" + str(i)]
            parameters["B" + str(i)] -= learning_rate * gradients["db" + str(i)]

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

    def evaluate(self, dataset):
        (X, Y) = dataset
        cache = self.forward_propagate(X, self.params, self.S, False)
        cost = MathUtils.cross_entropy(cache["A" + str(len(self.S) - 1)], Y)
        accuracy = self.get_accuracy(cache["A" + str(len(self.S) - 1)], Y)
        return {"cache": cache, "cost": cost, "accuracy": accuracy}

    # (784, 1) <--input data shape
    def predict(self, input_data):
        assert input_data.shape == (self.S[0], 1)
        X = input_data
        cache = self.forward_propagate(X, self.params, self.S, False)
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

    def save(self, path="neural_network"):
        outfile = open(path, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    @staticmethod
    def load(path="neural_network"):
        infile = open(path, 'rb')
        nn = pickle.load(infile)
        infile.close()
        return nn

    def gradient_check(self, parameters, gradients, X, Y, epsilon=1e-7):
        grad_approx = []
        grad = np.array([])
        count = 0
        # Preparing necessary analytic gradient values (only require dW and dB)
        for i in gradients.keys():
            if i.startswith("dw") or i.startswith("db"):
                new_vector = np.reshape(gradients[i], (-1, 1))
                if count == 0:
                    grad = new_vector
                else:
                    grad = np.concatenate((grad, new_vector), axis=0)
                count = count + 1
        grad = np.array(grad)

        # Building the numerical gradient approximations
        for i in parameters.keys():
            for idx in np.ndindex(parameters[i].shape):
                thetaplus = parameters[i][idx] + epsilon
                modified_params = parameters.copy()
                modified_params[i][idx] = thetaplus
                cache = self.forward_propagate(X, modified_params, self.S, True)
                J_Plus = MathUtils.cross_entropy(cache["A" + str(len(self.S) - 1)], Y)

                thetaminus = parameters[i][idx] - epsilon
                modified_params = parameters.copy()
                modified_params[i][idx] = thetaminus
                cache = self.forward_propagate(X, modified_params, self.S, True)
                J_Minus = MathUtils.cross_entropy(cache["A" + str(len(self.S) - 1)], Y)

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

    # (data, number_of_samples) <--input data shape ex:(784,60000) for mnist
    # [input_size,hidden_layers,output_size] <--shape list shape ex:[784,128,128,10] for mnist
    def __init__(self, dataset, val_dataset=None, epochs=2000, learning_rate=0.5, shape=None,
                 print_costs=True,
                 early_stopping=True, enable_dropout=True):
        self.enable_dropout = enable_dropout
        # Validations
        NN.validate_dataset(dataset, "Data set")
        if val_dataset is not None:
            NN.validate_dataset(val_dataset, "Validation set")

        (X, Y) = dataset
        assert len(X.shape) == 2
        assert len(Y.shape) == 2
        if val_dataset is not None:
            (X_test, Y_test) = val_dataset
            assert len(X_test.shape) == 2
            assert len(Y_test.shape) == 2

        if shape is not None:
            self.S = shape
            assert len(self.S) > 2
            assert self.S[0] == X.shape[0] and self.S[-1] == Y.shape[0]
        else:
            self.S = [dataset[0].shape[0], 30, 10]
        self.params = self.init_params(self.S)

        print("Starting training, NN dimensions: {}".format(self.S))

        previous_accuracy = 0
        previous_val_accuracy = 0

        for i in range(0, epochs):
            cache = self.forward_propagate(X, self.params, self.S, True)
            cost = MathUtils.cross_entropy(cache["A" + str(len(self.S) - 1)], Y)
            accuracy = self.get_accuracy(cache["A" + str(len(self.S) - 1)], Y)
            grads = self.back_propagate(X, Y, cache, self.params, self.S)
            #self.gradient_check(self.params, grads, X, Y)
            self.learn(grads, self.params, learning_rate, self.S)

            if i % 100 == 0:
                if print_costs:
                    print("Epoch:{},Cost:{}, Accuracy:{}".format(i, cost, accuracy))
                    if val_dataset is not None:
                        cache_test = self.forward_propagate(X_test, self.params, self.S, False)
                        cost_test = MathUtils.cross_entropy(cache_test["A" + str(len(self.S) - 1)], Y_test)
                        accuracy_test = self.get_accuracy(cache_test["A" + str(len(self.S) - 1)], Y_test)
                        print("Validation Cost:{}, Validation Accuracy:{}".format(cost_test, accuracy_test))
            if early_stopping:
                previous_accuracy = accuracy
                if val_dataset is not None:
                    previous_val_accuracy = accuracy_test
                    if previous_val_accuracy > accuracy_test:
                        break
                else:
                    if previous_accuracy > accuracy:
                        break