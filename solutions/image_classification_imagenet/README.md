# Image Classification of ImageNet database

This example shows how to use Convolutional Neural Networks to idenify an image using a pretrained model. The database of images we use is [ImageNet](http://www.image-net.org/). It consists on a database of 1000 classes where the objective is to classify objects in an image. The accuracy is meassured taking into account the error in the top 5 classes predicted. In 2015, Microsoft surpassed for the first time in history the human level performance in image classification using a CNN (see [He et al., 2015](http://arxiv.org/abs/1502.01852)).

## Download ImaneNet pretrained model 

To download ImageNet data:

       cd data
       chmod +x download_imagenet_data.sh
       ./download_imagenet_data.sh

## Run Convolutional Neural Networks

This example shows how to use a CNN to identify images using a pre-trained Inception-BatchNorm network. This configuratin is based in [Ioffe et al. (2015)](http://arxiv.org/abs/1502.03167v3).

* CNN with Inception-BatchNorm architecture:
    
        cd R/mxnet  
        R < inception_batchnorm.R --no-save  

