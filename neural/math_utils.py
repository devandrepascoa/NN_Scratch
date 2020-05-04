import numpy as np


def get_accuracy(Yhat, Y):
    """
    Function to calculate accuracy based on the model predictions and ground truth

    :param Yhat: Model prediction
    :param Y: Ground truth
    :return: accuracy
    """

    M = Yhat.shape[1]
    soma = 0
    Yhat_max = np.argmax(Yhat, axis=0)
    Y_max = np.argmax(Y, axis=0)
    for i in range(0, M):
        if Yhat_max[i] == Y_max[i]:
            soma += 1
    return (soma / M) * 100.0


def calculate_batches(X, Y, mini_batch_size, shuffle=True):
    M = X.shape[1]  # Number of training examples
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
        yield mini_batch_X, mini_batch_Y

    # In case there is one batch that is not complete yield it
    if M % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_batches * mini_batch_size:]
        mini_batch_Y = Y[:, num_complete_batches * mini_batch_size:]
        yield mini_batch_X, mini_batch_Y
    return X, Y


def hotOne(array, output_size):

    assert len(array.shape) == 2, "Input has to have shape (data,data_size)"
    Y_orig = array
    Y = np.zeros((output_size, Y_orig.shape[-1]))
    for i in range(0, Y_orig.shape[1]):
        value = Y_orig[0, i]
        Y[value, i] = 1.0
    return Y
