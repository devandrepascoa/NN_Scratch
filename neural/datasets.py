from keras.datasets import cifar10, mnist

from neural import math_utils
import numpy as np


def loadMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], -1)).T / 255.0
    x_test = (x_test.reshape(x_test.shape[0], -1)).T / 255.0
    y_train = (y_train.reshape(y_train.shape[0], 1)).T
    y_test = (y_test.reshape(y_test.shape[0], 1)).T
    y_train = math_utils.hotOne(y_train, 10)
    y_test = math_utils.hotOne(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def loadCifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.reshape(x_train.shape[0], -1) / 255.0).T
    x_test = (x_test.reshape(x_test.shape[0], -1) / 255.0).T
    y_train = (y_train.reshape(y_train.shape[0], 1)).T
    y_test = (y_test.reshape(y_test.shape[0], 1)).T
    y_train = math_utils.hotOne(y_train, 10)
    y_test = math_utils.hotOne(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def validate_dataset(dataset, name):
    assert isinstance(dataset, tuple) and len(dataset) == 2, (
            name + " has to be a tuple of size 2 -> (x_train,y_train)")
    assert isinstance(dataset[0], np.ndarray) and isinstance(dataset[1], np.ndarray), (
            name + " data has to be numpy array")
    assert len(dataset[0].shape) == 2 and len(dataset[1].shape) == 2, (
            name + " data has to be of shape (data_size,data) and labels (label_size,label)")
