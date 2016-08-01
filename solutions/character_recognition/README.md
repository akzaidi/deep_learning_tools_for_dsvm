# Character Recognition

This example shows how Convolutional Neural Networks work with character recognition.

## Download MNIST database

To download the MNIST database:

   cd data
   chmod +x download_mnist_data.sh
   ./download_mnist_data.sh

## Run Convolutional Neural Networks

There are two configurations for the CNN, one is a fully connected CNN and the other is the configuration proposed by [Lecun in 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). It is know as LeNet architecture.

* Fully connected CNN:
    
        cd R/mxnet  
        R < fully_connected.R --no-save  

* Lenet CNN:

        cd R/mxnet  
        R < lenet.R --no-save  