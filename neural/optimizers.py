import numpy as np


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def gradient_descent(self, gradients, parameters, learning_rate):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        parameters["W" + str(i)] -= learning_rate * gradients["dw" + str(i)]
        parameters["B" + str(i)] -= learning_rate * gradients["db" + str(i)]


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def gradient_descent_with_momentum(self, gradients, parameters, learning_rate):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["mdw" + str(i)] = self.Beta1 * self.directions["mdw" + str(i)] \
                                          + (1 - self.Beta1) * gradients["dw" + str(i)]
        self.directions["mdb" + str(i)] = self.Beta1 * self.directions["mdb" + str(i)] \
                                          + (1 - self.Beta1) * gradients["db" + str(i)]

        parameters["W" + str(i)] -= learning_rate * self.directions["mdw" + str(i)]
        parameters["B" + str(i)] -= learning_rate * self.directions["mdb" + str(i)]


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def RMS_Prop(self, gradients, parameters, learning_rate, epsilon=1e-8):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["rdw" + str(i)] = self.Beta2 * self.directions["rdw" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["dw" + str(i)], 2)
        self.directions["rdb" + str(i)] = self.Beta2 * self.directions["rdb" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["db" + str(i)], 2)

        parameters["W" + str(i)] -= learning_rate * gradients["db" + str(i)] / (
                np.sqrt(self.directions["rdw" + str(i)]) + epsilon)
        parameters["B" + str(i)] -= learning_rate * gradients["db" + str(i)] / (
                    np.sqrt(self.directions["rdb" + str(i)]) + epsilon)


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
# constant to prevent division by 0
def Adam_Optimizer(self, gradients, parameters, learning_rate, epsilon=1e-9):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["mdw" + str(i)] = self.Beta1 * self.directions["mdw" + str(i)] \
                                          + (1 - self.Beta1) * gradients["dw" + str(i)]
        self.directions["mdb" + str(i)] = self.Beta1 * self.directions["mdb" + str(i)] \
                                          + (1 - self.Beta1) * gradients["db" + str(i)]

        self.directions["rdw" + str(i)] = self.Beta1 * self.directions["rdw" + str(i)] \
                                          + (1 - self.Beta1) * np.power(gradients["dw" + str(i)], 2)
        self.directions["rdb" + str(i)] = self.Beta1 * self.directions["rdb" + str(i)] \
                                          + (1 - self.Beta1) * np.power(gradients["db" + str(i)], 2)

        parameters["W" + str(i)] -= learning_rate * self.directions["mdw" + str(i)] / (np.sqrt(
            self.directions["rdw" + str(i)]) + epsilon)
        parameters["B" + str(i)] -= learning_rate * self.directions["mdb" + str(i)] / (np.sqrt(
            self.directions["rdb" + str(i)]) + epsilon)
