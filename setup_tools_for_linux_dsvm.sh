#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
THIS_FOLDER=$PWD
INSTALL_FOLDER=installer
OPENBLAS_VERSION=0.2.18
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6
SKLEARN_VERSION=0.17.1
CHAINER_VERSION=1.12.0
TORCH_VERSION=2

# Create installation folder
mkdir ~/$INSTALL_FOLDER

# Install openblas
# FIXME: check if the version is the same, if nto, then update
echo "Installing open-blas version $OPENBLAS_VERSION"
OPENBLAS_FILE=/usr/local/include/openblas_config.h
if [ ! -e $OPENBLAS_FILE ] ; then
	cd ~/$INSTALL_FOLDER
	git clone --branch v$OPENBLAS_VERSION https://github.com/xianyi/OpenBLAS/
	cd OpenBLAS
	make FC=gfortran -j $(($(nproc) + 1))
	sudo make PREFIX=/usr/local install
	cd $THIS_FOLDER
fi

#Install theano
echo "Installing theano library version $THEANO_VERSION"
sudo `which pip` install theano==$THEANO_VERSION

# Install keras
echo "Installing keras library version $KERAS_VERSION"
sudo `which pip` install keras==$KERAS_VERSION

# Install scikit-learn
echo "Installing scikit-learn library version $SKLEARN_VERSION"
sudo `which pip` install scikit-learn==$SKLEARN_VERSION

# Install chainer
echo "Installing chainer library version $CHAINER_VERSION"
sudo `which pip` install chainer==$CHAINER_VERSION

# Install torch
echo "Installing torch library version $TORCH_VERSION"
sudo yum install -y readline-devel ncurses-devel libjpeg-turbo-devel libpng-devel GraphicsMagick-devel fftw-devel sox-devel sox qt-devel qtwebkit-devel 
TORCH_FILE=/usr
cd ~/$INSTALL_FOLDER
git clone https://github.com/torch/distro.git torch --recursive
cd torch
cd $THIS_FOLDER

# Install mxnet

# Install caffe




