# Generative Adversarial Networks

In this folder there is a colection of generative adversarial network (GAN) examples. GANs are considered by some of the
top deep learning scientists the most important advance for the field in the last 10 years. See this [quora answer]
(https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning) by
LeCun. Specifically they are the most important advance in unsupervised learning.

GAN were first proposed in 2014 by [Goodfellow et al](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). Here you will find a complete [explanation](https://code.facebook.com/posts/1587249151575490/a-path-to-unsupervised-learning-through-adversarial-networks).
 
NOTE: to run this example you will have to install [opencv](http://opencv.org/).


## Download database

To download the MNIST database:

    cd data
    python download_mnist_data.sh

## Run Generative Adversarial Networks

To run an example of GAN using MNIST character recognotions database: 

* GAN with MNIST semisupervised:
    
    cd python/mxnet  
    python gan_mnist_semisupervised.py
    