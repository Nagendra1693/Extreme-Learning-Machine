import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import random


def get_data(source):
    if source == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = np.dot(x_train, [0.299, 0.587, 0.114])
        x_test = np.dot(x_test, [0.299, 0.587, 0.114])
        x_train = x_train.reshape(x_train.shape[0], 32 * 32)
        x_test = x_test.reshape(x_test.shape[0], 32 * 32)
    elif source == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28 * 28)
        x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    else:
        return None
    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)