import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from .layers import ParamMixin
from .helpers import one_hot, unhot
import os
class NeuralNetwork:
    """
    A neural network class that allows definition and training of a
    neural network using stochastic gradient descent. Supported layer
    types are Linear (InnerProduct), Convolution, Pooling. Supported
    activation functions include Sigmoid, Tanh and ReLU. Softmax
    cross-entropy loss (classification loss) is used at the output
    layer for training.
    """

    def __init__(self, layers, rng=None):
        """
        Initialize the neural network.
        :param layers: list of Layer objects
        :param rng: (optional) np.random.RandomState container
        :return:
        """
        # layers is a list of layer params
        self.layers = layers

        # initialize random number generator
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        # store loss / error
        self.tr_error = []
        self.te_error = []
        self.tr_loss = []
        self.conf_mat = None

        # figure handles for plotting
        self.filter = plt.figure(1)
        self.fig_loss = plt.figure(2)
        self.fig_confmat = plt.figure(3)
        
    def _setup(self, X, Y):
        """
        Setup layers sequentially.
        """
        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))

    def fit(self, X, Y, X_test, Y_test, test_iter=2, learning_rate=0.1, max_iter=10, batch_size=64,cnn=True):
        """
        Train a network using stochastic gradient descent with mini-batches.
        :param X: Train data
        :param Y: Train labels -
        :param X_test: Test data
        :param Y_test: Test labels
        :param test_iter: Evaluate network on test set every test_iter iterations
        :param learning_rate: learning rate
        :param max_iter: Total number of iterations of SGD to run
        :param batch_size: Batch size for SGD
        :param cnn: True if network is a cnn
        :return: Nothing
        """
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size

        # Convert output label (scalar) values to a one-hot vector to make it compatible with the loss layer
        # implementation
        Y_one_hot = one_hot(Y)

        # Setup layers
        self._setup(X, Y_one_hot)

        # Stochastic gradient descent with mini-batches using back prop algorithm
        iter = 0
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                # Create mini-batch
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y_one_hot[batch_begin:batch_end]

                # Forward propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.fprop(X_next)
                Y_pred = X_next

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, ParamMixin):
                        for param, inc in zip(layer.params(),
                                              layer.param_incs()):
                            param -= learning_rate*inc

                print('epoch %i iter %i ' % (iter, b))

            # Output training status
            loss = self._loss(X, Y_one_hot)
            error = self.error(X, Y)

            # Store loss / error on train
            self.tr_error.append((iter,error))
            self.tr_loss.append((iter,loss))
            print('epoch %i, loss %.4f, train error %.4f' % (iter, loss, error))

            # Compute test error and confusion matrix
            error, cm = self.error(X_test, Y_test, conf=True)
            self.conf_mat = cm
            self.te_error.append((iter,error))
            print('epoch %i, test error %.4f' % (iter, error))
            
            # Visualize the filters
            if cnn:
                W,b = self.layers[0].params()           
                self.vis_square(W.transpose(1,2,3,0),iter)
            else:
                W,b = self.layers[1].params()
                # np.save(str(iter), W)
                self.plot_rf(W.copy(), iter)

            # Plot loss
            self.call_plot(blockFig=False)
                
    def _loss(self, X, Y_one_hot):
        """ Computes the loss at the output layer. """
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_one_hot, Y_pred)

    def predict(self, X):
        """
        Calculate an output Y for the given input X using forward propagation.
        Output y in Y is the class with max value at the output layer.
        :param X: NxD ndarray containing D dimensional input
        :return: N vector of predicted labels
        """
        # Forward propagate input X through all the layers
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        # X_next has the output layer values
        # Unhot X_next to get output class labels
        Y_pred = unhot(X_next)
        return Y_pred

    def error(self, X, Y, conf=False):
        """
        Calculate classification error on the given data.
        :param X: NxD ndarray containing D dimensional input
        :param Y: size N vector containing true output class labels for input
        :param conf: Compute confusion matrix is True
        :return: error, conf mat
        """
        Y_pred = self.predict(X)
        error = Y_pred != Y
        if conf:
            cm = confusion_matrix(Y, Y_pred)
            return np.mean(error), cm
        return np.mean(error)

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        # Warning: the following is a hack
        Y_one_hot = one_hot(Y)
        self._setup(X, Y_one_hot)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                print('layer %d' % l)
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape

                    def fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y_one_hot)

                    def grad_fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation
                        X_next = X
                        for layer in self.layers:
                            X_next = layer.fprop(X_next)
                        Y_pred = X_next

                        # Back-propagation of partial derivatives
                        next_grad = self.layers[-1].input_grad(Y_one_hot,
                                                               Y_pred)
                        for layer in reversed(self.layers[l:-1]):
                            next_grad = layer.bprop(next_grad)
                        return np.ravel(self.layers[l].param_grads()[p])

                    param_init = np.ravel(np.copy(param))
                    err = sp.optimize.check_grad(fun, grad_fun, param_init)
                    print('diff %.2e' % err)

    # def call_plot(self, blockFig=False):
        # """
          # Plots the error curves and the test confusion matrix.
        # """
        # self.fig_confmat.clf()
        # ax_loss = self.fig_loss.add_subplot(1,1,1)
        # ax_confmat = self.fig_confmat.add_subplot(1,1,1)
        # ax_loss.plot([i[0] for i in self.tr_error], [i[1] for i in self.tr_error], 'r')
        # ax_loss.plot([i[0] for i in self.te_error], [i[1] for i in self.te_error], 'b')
        # cm = self.conf_mat
        # if cm is not None:
            # cm1 = np.asarray(cm).astype(np.float)
            # for i,r in enumerate(cm1):
                # cm1[i,:] = (cm1[i,:]/cm1[i,:].sum())*100.0
            
            # cax = ax_confmat.matshow(cm)
            # self.fig_confmat.colorbar(cax)
            # ax_confmat.set_title('Test Confusion matrix')
            # ax_confmat.set_ylabel('True label')
            # ax_confmat.set_xlabel('Predicted label')
            # plt.draw()
            # if blockFig:
                # plt.show(block=True)
            # else:
    #             plt.show(block=False)
    def call_plot(self, blockFig=False, save_folder = None):
        """
          Plots the error curves and the test confusion matrix.
        """
        self.fig_confmat.clf()
        ax_loss = self.fig_loss.add_subplot(1,1,1)
        ax_confmat = self.fig_confmat.add_subplot(1,1,1)
        ax_loss.plot([i[0] for i in self.tr_error], [i[1] for i in self.tr_error], 'r')
        ax_loss.plot([i[0] for i in self.te_error], [i[1] for i in self.te_error], 'b')
        cm = self.conf_mat
        if cm is not None:
            cm1 = np.asarray(cm).astype(np.float)
            for i,r in enumerate(cm1):
                cm1[i,:] = (cm1[i,:]/cm1[i,:].sum())*100.0
            
            cax = ax_confmat.matshow(cm)
            self.fig_confmat.colorbar(cax)
            ax_confmat.set_title('Test Confusion matrix')
            ax_confmat.set_ylabel('True label')
            ax_confmat.set_xlabel('Predicted label')
            plt.draw()
            if blockFig:
                path_to_save = os.path.join(os.path.curdir,
                                            'Result',save_folder) 
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                self.fig_loss.savefig(os.path.join(path_to_save, 'loss.png'))
                self.fig_confmat.savefig(os.path.join(path_to_save,'confusion_matrix.png'))
                plt.show(block=True)
            else:
                plt.show(block=False)

            
    def vis_square(self, data, iter, padsize=1, padval=0):
        """
        Visualizes the weights of the first conv layer of a CNN.
        """
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        if data.shape[2] == 1:
            data = np.tile(data,3)
        ax = self.filter.add_subplot(111)
        ax.imshow(data, cmap=plt.get_cmap('jet'))
        # self.filter.savefig('weights_'+str(iter)+'.png')
        plt.draw()
        plt.show(block=False)
        
    def plot_rf(self, w1, iter_num, lim=0.1):
        """
         Visualizes the weights of the the first hidden layer in a fully connected network.
         For each neuron in the first hidden layer, the corresponding weight vector is
         reshaped to a square and tiled onto an weight image.
        """
        w = w1.transpose(1, 0)
        lim = w.std()*2.

        N1 = int(np.sqrt(w.shape[1]))           # sqrt(784) = 28, you have 28x28 RFs
        N2 = int(np.ceil(np.sqrt(w.shape[0])))  # sqrt(256) = 16, you have 16x16 output cells
        
        W = np.zeros((N1*N2,N1*N2))             # You are creating a weight wall of 16x16 RF blocks
        
        for j in range(w.shape[0]):
            r = int(j/N2)
            c = int(j%N2)
            x = c*N1
            y = r*N1
            W[y:y+N1, x:x+N1] = w[j, :].reshape((N1, N1))

        self.filter.clf()
        ax = self.filter.add_subplot(1,1,1)
        ax.imshow(W, vmin=-lim, vmax=lim, cmap=plt.get_cmap('gray'))
        self.filter.savefig('weights_' + str(iter_num) + '.png')
        plt.draw()
        plt.show(block=False)
