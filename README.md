# DL Workshop - Neural Net

This code is meant as an easy to read implementation to understand neural network internals and for small experiments.
Meant to run on toy problems such as MNIST.

Modules included:

1. Fully Connected / Convolutional Feed-Forward network
2. Restricted Boltzmann Machine

## Build
A small part of the code is in Cython, which requires a build. 
Use the relevant way to install dependencies and build the code.

For windows users, you will need to install Cython with MingGW or Visual C++ in order to build the code (option 1). 
See https://github.com/cython/cython/wiki/InstallingOnWindows for Cython installation on windows. 

Alternately, a vagrant linux VM (option 3) has been included here, which can be run using virtualbox.

Option 1 (Using a virtualenv and pip to install dependencies):

    pip install virtualenv
    pip install --upgrade pip
    git clone https://github.com/MadStreetDen-OpenSource/dlworkshop5el.git 
    cd dlworkshop5el
    virtualenv dlw
    source dlw/bin/activate
    pip install -r requirements.txt
    python setup.py build

Option 2 (Using apt-get to install dependencies - Linux):

    sudo apt-get install python-dev
    sudo apt-get install python-numpy
    sudo apt-get install cython
    sudo apt-get install python-scipy
    sudo apt-get install python-matplotlib
    sudo apt-get install python-sklearn
    sudo apt-get install python-setuptools
    
    python setup.py build

Option 3(Vagrant Virtual Machine), (Recommended for people having Windows machines):
Disable secure boot. And enable Virtualization in Boot Setup.

    1. Get an image of vagrant from the vagrant website.
    2. Install a virtualization tool. virtualbox is recommended.
    3. Clone repository and cd into dlworkshop5el
    4. Run command $ vagrant up
    6. Open virtualbox. You can see the a new virtual machine running.
    7. Click on that virtual machine and on the top click show.
    8. A new screen will pop up. Login into it using useraname: vagrant and password vagrant.
    9. Run command $ cd /vagrant
    5. Run command $ python setup.py build

This should create build folder in the root folder. 

Tip: To get your mouse pointer back after clicking on virtualbox window press Right ctrl key.

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
