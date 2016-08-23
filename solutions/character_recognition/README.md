# Character Recognition

This example shows how Convolutional Neural Networks work with character recognition.

## Download MNIST database

To download the MNIST database:

    cd data
    python download_mnist_data.sh

## Run Convolutional Neural Networks

There are two configurations for the CNN, one is a fully connected CNN and the other is the configuration proposed by [Lecun in 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). It is know as LeNet architecture.

* Fully connected CNN:
    
    cd R/mxnet  
    R < fully_connected.R --no-save  

* Lenet CNN in R
Scripts in R with simple and advanced methods with mxnet:

    cd R/mxnet  
    R < lenet_simple.R --no-save  
    R < lenet_advanced.R

* Lenet CNN in python
Notebook with a simple and an advanced methods in python with mxnet:

	cd python/mxnet
	jupiter notebook 