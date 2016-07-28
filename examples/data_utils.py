import sklearn.datasets
import numpy as np
import math
import os
import matplotlib.pyplot as plt


def load_data_mnist():
    # loads the mnist dataset
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home=os.path.join(os.path.dirname(__file__),'../data'))
    data = mnist.data/255.
    target = mnist.target
    return data, target


def load_data_2d(dataset, n_data=1000):
    # returns 2d synthetically generated data
    if dataset == 'circle':
        data, target = gen_circle_data(n_data)
    elif dataset == 'spiral':
        data, target = gen_spiral_data(n_data)
    elif dataset == 'random':
        data, target = gen_random_data(n_data)
    # plot_data(data, target)
    return data, target


def gen_circle_data(n_data):
    n_per_class = n_data / 2
    r1 = np.random.uniform(0, 0.4, (n_per_class))
    r2 = np.random.uniform(0.6, 1.0, (n_per_class))
    r = np.concatenate((r1,r2), axis=0)
    t = np.random.uniform(0, 2*math.pi, (n_data))
    data = np.asarray([[x*math.cos(y), x*math.sin(y)] for x,y in zip(r,t)])
    target = np.concatenate((np.zeros((n_per_class), dtype=np.int32), np.ones((n_per_class), dtype=np.int32)))

    return data, target


def gen_spiral_data(n_data):
    n_per_class = n_data / 2

    r0 = 5.*np.arange(0, 1, 1./n_per_class) + np.random.uniform(-0.1, 0.1, (n_per_class))
    t0 = 1.25*2*np.pi*np.arange(0, 1, 1./n_per_class) + np.random.uniform(-0.1, 0.1, (n_per_class))

    r1 = 5.*np.arange(0, 1, 1./n_per_class) + np.random.uniform(-0.1, 0.1, (n_per_class))
    t1 = 1.25*2*np.pi*np.arange(0, 1, 1./n_per_class) + np.random.uniform(-0.1, 0.1, (n_per_class))
    t1 += np.pi

    r = np.concatenate((r0, r1), axis=0)
    t = np.concatenate((t0, t1), axis=0)

    data = np.zeros((r.shape[0], 2))
    data[:, 0] = r * np.cos(t)
    data[:, 1] = r * np.sin(t)

    target = np.concatenate((np.zeros((n_per_class), dtype=np.int32), np.ones((n_per_class), dtype=np.int32)))

    return data, target


def gen_random_data(n_data):
    data = np.random.uniform(-5, 5, (n_data, 2))
    target = np.concatenate((np.zeros((n_data/2), dtype=np.int32), np.ones((n_data/2), dtype=np.int32)))
    return data, target


def plot_data(data, target):
    idx_0 = np.argwhere(target==0)
    idx_0 = idx_0.reshape((idx_0.shape[0]))
    idx_1 = np.argwhere(target==1)
    idx_1 = idx_1.reshape((idx_1.shape[0]))

    plt.figure()
    plt.plot(data[idx_0,0], data[idx_0,1], 'rx')
    plt.plot(data[idx_1,0], data[idx_1,1], 'go')
    plt.show()


def shuffle_data(data, target):
    N = data.shape[0]
    idx = np.random.permutation(N)
    data = data[idx]
    target = target[idx]

    return data, target


if __name__ == "__main__":
    load_data_2d('circle', 200)