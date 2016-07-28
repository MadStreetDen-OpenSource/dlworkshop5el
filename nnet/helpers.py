import numpy as np

"""
This file defines helper functions:
one_hot
unhot

Activation functions and their derivatives -

sigmoid, sigmoid_d
tanh, tanh_d
relu, relu_d

"""

def one_hot(labels):
    """
    Convert each label in labels into a num_classes sized one hot vector representation.
    Returns an ndarray of size labels.shape x num_classes containing one hot vectors.
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    """ Computes scalar class labels by taking argmax along the innermost dimension """
    return np.argmax(one_hot_labels, axis=-1)


def sigmoid(x):
    """ Compute sigmoid activation for input x """
    return 1.0/(1.0+np.exp(-x))


def sigmoid_d(x):
    """ Compute derivative of the sigmoid activation fn for input x """
    s = sigmoid(x)
    return s*(1-s)


def tanh(x):
    """ Compute tanh activation for input x """
    return np.tanh(x)


def tanh_d(x):
    """ Compute derivative of the tanh activation fn for input x """
    e = np.exp(2*x)
    return (e-1)/(e+1)


def relu(x):
    """ Compute relu (Rectified Linear unit) activation for input x """
    return np.maximum(0.0, x)


def relu_d(x):
    """ Compute derivative of the relu activation fn for input x """
    dx = np.zeros(x.shape)
    dx[x >= 0] = 1
    return dx
