#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
HOME_USER=$HOME
sudo su
HOME_SUDO=$HOME
THIS_FOLDER=$PWD
INSTALL_FOLDER=/tmp/installer
OPENBLAS_VERSION=0.2.18
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6
SKLEARN_VERSION=0.17.1
CHAINER_VERSION=1.12.0
TORCH_VERSION=LUAJIT21

# Create installation folder
mkdir -p $INSTALL_FOLDER

# Install openblas
# FIXME: check if the version is the same, if nto, then update
echo "Installing OpenBLAS version $OPENBLAS_VERSION"
OPENBLAS_FILE=/usr/local/include/openblas_config.h
if [ ! -e $OPENBLAS_FILE ] ; then
	cd ~/$INSTALL_FOLDER
	git clone --branch v$OPENBLAS_VERSION https://github.com/xianyi/OpenBLAS/
	cd OpenBLAS
	make FC=gfortran -j $(($(nproc) + 1))
	make PREFIX=/usr/local install
	cd $THIS_FOLDER
else
	echo "WARNING: OpenBLAS already installed"
fi

#Install theano
echo "Installing theano library version $THEANO_VERSION"
`which pip` install theano==$THEANO_VERSION

# Install keras
echo "Installing keras library version $KERAS_VERSION"
`which pip` install keras==$KERAS_VERSION

# Install scikit-learn
echo "Installing scikit-learn library version $SKLEARN_VERSION"
`which pip` install scikit-learn==$SKLEARN_VERSION

# Install chainer
echo "Installing chainer library version $CHAINER_VERSION"
`which pip` install chainer==$CHAINER_VERSION

# Install torch
# FIXME: check if it is already installed, if not, then do it
echo "Installing torch library version $TORCH_VERSION"
TORCH_FILE=th
if [! command -v $TORCH_FILE]; then
	yum install -y cmake readline-devel ncurses-devel libjpeg-turbo-devel libpng-devel GraphicsMagick-devel fftw-devel sox-devel sox qt-devel qtwebkit-devel 
	cd $INSTALL_FOLDER
	git clone https://github.com/torch/distro.git torch --recursive
	cd torch
	TORCH_LUA_VERSION=$TORCH_VERSION ./install.sh -b
	source $HOME_USER/.bashrc
	cd $THIS_FOLDER
fi

# Install caffe
#echo "Installing caffe library version $CAFFE_VERSION"
#yum install -y libleveldb-dev 
#yum install --no-install-recommends libboost-all-dev

# Install mxnet



