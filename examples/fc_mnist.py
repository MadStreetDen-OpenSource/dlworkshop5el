#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
import nnet
import data_utils

"""
This example classifies the MNIST digit recognition data
using a fully connected network.
"""

def run():
    # Fetch MNIST data
    data, target = data_utils.load_data_mnist()
    split = 60000

    # Split data into train and test data
    X_train = data[:split, :]
    y_train = target[:split]
    n_classes = np.unique(y_train).size

    X_test = data[split:, :]
    y_test = target[split:]

    # Shuffle the training set
    X_train, y_train = data_utils.shuffle_data(X_train, y_train)

    # Setup fully connected neural network
    nn = nnet.NeuralNetwork(
        # layers is a list of Layer objects
        # A layer object can be implemented as different types - Linear, Conv etc.
        layers=[
            # flatten the 32x32 input image into a 1024 dim vector
            nnet.Flatten(),
            # fully connected layer #1
            nnet.Linear(
                n_out=64,          # no. of outputs = 256
                weight_scale=0.0001,   # random initialization from normal distribution - scale
                weight_decay=0.0,  # weight decay factor
            ),
            # relu activation for fc1
            nnet.Activation('relu'),
            # fully connected layer #2
            nnet.Linear(
                n_out=32,
                weight_scale=0.1,
                weight_decay=0.0,
            ),
            # relu activation for fc2
            nnet.Activation('relu'),
            # fully connected layer #3
            nnet.Linear(
                n_out=n_classes,    # last layer has no. of outputs = no. of classes
                weight_scale=0.1,
                weight_decay=0.0,
            ),
            # relu activation for fc3
            nnet.Activation('relu'),
            # Softmax cross-entropy loss layer
            nnet.LogRegression(),
        ],
    )

    # Train neural network using SGD
    # max_iter = number of epochs
    # batch_size = batch size for SGD
    # test_iter = no. of epochs to test after
    t0 = time.time()
    nn.fit(X_train, y_train, X_test, y_test, learning_rate=0.001, max_iter=50, batch_size=100, test_iter=10, cnn=False)
    t1 = time.time()
    print('Training Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)

    # plot loss curve
    nn.call_plot(blockFig=True)

    # Classify 10 random images from the test set
    idx = np.random.permutation(X_test.shape[0])[:10]
    for i in idx:
        print nn.predict(X_test[i].reshape(1,28*28))
        plt.figure(); plt.imshow(X_test[i].reshape(28,28)); plt.show()

if __name__ == '__main__':
    run()