import numpy as np


# Function to initialize our parameters( Weights and biases)
def init_params(self):
    parameters = dict()
    length = len(self.S)  # neural network length
    for i in range(1, length):
        # (np.sqrt(2 / S[i - 1])) important for weight initialization, using He initialization(NOT Xavier)
        parameters["W" + str(i)] = np.random.randn(self.S[i], self.S[i - 1]) * (np.sqrt(2 / self.S[i - 1]))
        parameters["B" + str(i)] = np.zeros((self.S[i], 1))  # bias can be started with zero, doesn't affect much
    return parameters


# Function to initialize directions for
def init_params_momentum(self):
    directions = dict()
    length = len(self.S)  # neural network length
    for i in range(1, length):
        directions["mdw" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
        directions["mdb" + str(i)] = np.zeros(self.params["B" + str(i)].shape)
    return directions


# Function to initialize directions for rms prop
def init_params_rms(self):
    directions = dict()
    length = len(self.S)  # neural network length
    for i in range(1, length):
        directions["rdw" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
        directions["rdb" + str(i)] = np.zeros(self.params["B" + str(i)].shape)
    return directions


# Function to initialize directions for adam
def init_params_adam(self):
    directions = dict()
    length = len(self.S)  # neural network length
    for i in range(1, length):
        directions["mdw" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
        directions["mdb" + str(i)] = np.zeros(self.params["B" + str(i)].shape)
        directions["rdw" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
        directions["rdb" + str(i)] = np.zeros(self.params["B" + str(i)].shape)

    return directions
