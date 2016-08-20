# Image Classification of CIFAR database

This group of examples shows how to use Convolutional Neural Networks to identify classes in an image using the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) database. The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. 

This example shows ResNet architecture to model the network, other architectures in R and python can be found [here](https://github.com/dmlc/mxnet/tree/master/example/image-classification). [ResNet](http://arxiv.org/abs/1512.03385) paper was published in December 2015 by the Microsoft Research team. It was the same team that in February 2015 surpassed the [human-level performance](http://arxiv.org/abs/1502.01852) on ImageNet classification contest. This is a good example of the inmense power that deep learning can achieve in computer vision problems.

## Download CIFAR-10 dataset 

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. To download the data:

	cd data
	python download_data.py

## Run Convolutional Neural Networks

ResNet have a depth of 152 convolutional layers. This examples have two versions a small one and the full one.

* Execution in R with mxnet library:

		cd R/mxnet
		R < train_resnet.R --no-save  

* Execution in python with mxnet library:
    
    	cd python/mxnet  
    	python train_resnet.py  

