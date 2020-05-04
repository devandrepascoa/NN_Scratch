import numpy as np


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def gradient_descent(self, gradients, parameters, learning_rate):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        parameters["W" + str(i)] -= learning_rate * gradients["dw" + str(i)]
        parameters["B" + str(i)] -= learning_rate * gradients["db" + str(i)]


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def gradient_descent_with_momentum(self, epoch, gradients, parameters, learning_rate, threshold=5.0):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["mdw" + str(i)] = self.Beta1 * self.directions["mdw" + str(i)] \
                                          + (1 - self.Beta1) * gradients["dw" + str(i)]
        self.directions["mdb" + str(i)] = self.Beta1 * self.directions["mdb" + str(i)] \
                                          + (1 - self.Beta1) * gradients["db" + str(i)]

        mdw_corrected = self.directions["mdw" + str(i)] / (1 - np.power(self.Beta1, epoch))
        mdb_corrected = self.directions["mdb" + str(i)] / (1 - np.power(self.Beta1, epoch))

        parameters["W" + str(i)] -= learning_rate * mdw_corrected
        parameters["B" + str(i)] -= learning_rate * mdb_corrected

        parameters["W" + str(i)] = parameters["W" + str(i)] * (threshold / np.linalg.norm(parameters["W" + str(i)]))
        parameters["B" + str(i)] = parameters["B" + str(i)] * (threshold / np.linalg.norm(parameters["B" + str(i)]))


# Function to adjust our weights and biases(parameters) based on the respective gradient and learning rate
def RMS_Prop(self, epoch, gradients, parameters, learning_rate, threshold=5.0, epsilon=1e-8):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["rdw" + str(i)] = self.Beta2 * self.directions["rdw" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["dw" + str(i)], 2)
        self.directions["rdb" + str(i)] = self.Beta2 * self.directions["rdb" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["db" + str(i)], 2)

        rdw_corrected = self.directions["rdw" + str(i)] / (1 - np.power(self.Beta2, epoch))
        rdb_corrected = self.directions["rdb" + str(i)] / (1 - np.power(self.Beta2, epoch))

        delta_W = gradients["db" + str(i)] / (np.sqrt(rdw_corrected) + epsilon)
        delta_B = gradients["db" + str(i)] / (np.sqrt(rdb_corrected) + epsilon)

        parameters["W" + str(i)] -= learning_rate * delta_W
        parameters["B" + str(i)] -= learning_rate * delta_B

        parameters["W" + str(i)] = parameters["W" + str(i)] * (threshold / np.linalg.norm(parameters["W" + str(i)]))
        parameters["B" + str(i)] = parameters["B" + str(i)] * (threshold / np.linalg.norm(parameters["B" + str(i)]))


def Adam_Optimizer(self, epoch, gradients, parameters, learning_rate, threshold=5.0, epsilon=1e-9):
    length = len(self.S)  # neural network length
    for i in range(1, length):
        self.directions["mdw" + str(i)] = self.Beta1 * self.directions["mdw" + str(i)] \
                                          + (1 - self.Beta1) * gradients["dw" + str(i)]
        self.directions["mdb" + str(i)] = self.Beta1 * self.directions["mdb" + str(i)] \
                                          + (1 - self.Beta1) * gradients["db" + str(i)]

        self.directions["rdw" + str(i)] = self.Beta2 * self.directions["rdw" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["dw" + str(i)], 2)
        self.directions["rdb" + str(i)] = self.Beta2 * self.directions["rdb" + str(i)] \
                                          + (1 - self.Beta2) * np.power(gradients["db" + str(i)], 2)

        mdw_corrected = self.directions["mdw" + str(i)] / (1 - np.power(self.Beta1, epoch))
        mdb_corrected = self.directions["mdb" + str(i)] / (1 - np.power(self.Beta1, epoch))

        rdw_corrected = self.directions["rdw" + str(i)] / (1 - np.power(self.Beta2, epoch))
        rdb_corrected = self.directions["rdb" + str(i)] / (1 - np.power(self.Beta2, epoch))

        delta_W = mdw_corrected / (np.sqrt(rdw_corrected) + epsilon)
        delta_B = mdb_corrected / (np.sqrt(rdb_corrected) + epsilon)

        parameters["W" + str(i)] -= learning_rate * delta_W
        parameters["B" + str(i)] -= learning_rate * delta_B

        parameters["W" + str(i)] = parameters["W" + str(i)] * (threshold / np.linalg.norm(parameters["W" + str(i)]))
        parameters["B" + str(i)] = parameters["B" + str(i)] * (threshold / np.linalg.norm(parameters["B" + str(i)]))
