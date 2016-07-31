#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import numpy as np
import matplotlib
# either of the two lines should be uncommented
# matplotlib.use('Agg') # uncomment this if you are using the VM, gives text
                        # ouptut only
# matplotlib.use('TkAgg') # uncomment this if you are using installed version
                          #gives visual output
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
import nnet
import data_utils

"""
This example classifies the MNIST digit recognition data
using a convolutional neural network.
"""

def run():
    # Fetch MNIST data
    data, target = data_utils.load_data_mnist()
    data = data.reshape(data.shape[0], 1, 28, 28)
    split = 60000

    # Split data into train and test data
    X_train = data[:split]
    y_train = target[:split]
    n_classes = np.unique(y_train).size

    X_test = data[split:]
    y_test = target[split:]

    # Shuffle the training set
    X_train, y_train = data_utils.shuffle_data(X_train, y_train)

    # Reduce the size of the training set (reducing training time)
    X_train, y_train = X_train[:5000], y_train[:5000]

    # Setup convolutional neural network
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            nnet.Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Flatten(),
            nnet.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02,
            ),
            nnet.LogRegression(),
        ],
    )

    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, X_test, y_test, learning_rate=0.01, max_iter=20, batch_size=100, test_iter=5)
    t1 = time.time()
    print('Training Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)

    # Plot loss curve
    nn.call_plot(blockFig=True, save_folder = 'cnn_mnist')

    # Classify 10 random images from the test set
    idx = np.random.permutation(X_test.shape[0])[:10]
    for i in idx:
        print nn.predict(X_test[i].reshape(1,1,28,28))
        plt.figure(); plt.imshow(X_test[i].reshape(28,28)); plt.show()


if __name__ == '__main__':
    run()
