# Deep Learning tools for the Data Science Virtual Machine

This repo will help you to leverage deep learning applications using the Microsoft's [Data Science Virtual Machine](https://azure.microsoft.com/en-gb/documentation/articles/machine-learning-data-science-linux-dsvm-intro/) (DSVM). It contains two kinds of resources: installation scripts to easily configure the best deep learning libraries in your DSVM and solution packages to let you start with deep learning applications. 

## Deep Learning Libraries
The following libraries are installed via a simple script:
* [CNTK](https://www.cntk.ai/): Microsoft's native library for deep learning (already preinstalled in the DSVM). Code in C# and C++.
* [mxnet](https://github.com/dmlc/mxnet): Distributed library for deep learning. Code in R, python, C++ and Scala.
* [keras](https://github.com/fchollet/keras): Popular deep learning library. Code in python.
* [caffe](https://github.com/BVLC/caffe): Deep learning library from Berkeley. Code in C++.
* [torch](https://github.com/torch/torch7): Mathematical framework for deep learning. Code in Lua and C.
* [chainer](https://github.com/pfnet/chainer): Deep learning library based in caffe. Code in python.
* [theano](https://github.com/Theano/Theano): Mathematical framework for deep learning. Code in python

To install them all, just log in in your DSVM (in Linux) and write:

    sudo chmod +x -R install.sh config
    ./install.sh

## Solutions
Next there are several solutions that will help you learn and understand deep learning.

* Character Recognition
* Image classification CIFAR
* Image classification ImageNet
* Implementation of artistic style in images
* Word prediction from characters
* Sentiment Analisys
