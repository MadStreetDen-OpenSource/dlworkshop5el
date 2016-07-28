# DL Workshop - Neural Net

This code is meant as an easy to read implementation to understand neural network internals and for small experiments.
Meant to run on toy problems such as MNIST.

Modules included:

1. Fully Connected / Convolutional Feed-Forward network
2. Restricted Boltzmann Machine

## Build

Steps to run programs
Option 1(both OS X and linux):

    pip install virtualenv
    git clone the repository using the command git clone 
    cd into the repository
    virtualenv dlw
    source dlw/bin/activate
    pip install -r requirements.txt
    python setup.py build

Option 2(linux):

    sudo apt-get install python-dev
    sudo apt-get install python-numpy
    sudo apt-get install cython
    sudo apt-get install python-scipy
    sudo apt-get install python-matplotlib
    python setup.py build

Option 3(Vagrant Virtual Machine), (Recommended for people having Windoze machines):
Disable secure boot. And enable Virtualization in Boot Setup

1. Get an image of vagrant from the vagrant website.
2. Install a virtualization tool. virtualbox is recommended.
3. change your Currrent working directory to the clone repository
4. Run command $vagrant up
5. Run command $python setup.py build

This should create build folder in the root folder.

## Examples

1. Fully connected network
This example classifies the MNIST digit recognition data using a fully connected network.
```
python examples/fc_mnist.py
```

2. CNN network
This example classifies the MNIST digit recognition data using a convolutional neural network.
```
python examples/cnn_mnist.py
```

3. RBM
This example trains multiple layers of RBMs and visualizes the reconstructed input.
```
python examples/rbm_multiple_layers_mnist.py
```

4. RBM + SVM 
This example trains multiple layers of RBMs and classifies the lower dimensional output of the RBM using an SVM.
```
python rbm_svm_mnist.py
```
