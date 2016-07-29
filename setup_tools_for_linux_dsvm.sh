#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
HOME_USER=$HOME
THIS_FOLDER=$PWD
INSTALL_FOLDER=/tmp/installer
OPENBLAS_VERSION=0.2.18
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6
SKLEARN_VERSION=0.17.1
CHAINER_VERSION=1.12.0
TORCH_VERSION=LUAJIT21
CAFFE_VERSION=rc3

# Create installation folder
mkdir -p $INSTALL_FOLDER

# Install openblas
# FIXME: check if the version is the same, if not, then update
echo "Installing OpenBLAS version $OPENBLAS_VERSION"
OPENBLAS_FILE=/usr/local/include/openblas_config.h
if [ ! -e $OPENBLAS_FILE ] ; then
	cd ~/$INSTALL_FOLDER
	git clone --branch v$OPENBLAS_VERSION https://github.com/xianyi/OpenBLAS/
	cd OpenBLAS
	make FC=gfortran -j $(($(nproc) + 1))
	sudo make PREFIX=/usr/local install
	cd $THIS_FOLDER
else
	echo "WARNING: OpenBLAS already installed"
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
sudo yum install -y cmake readline-devel ncurses-devel libjpeg-turbo-devel libpng-devel GraphicsMagick-devel fftw-devel sox-devel sox qt-devel qtwebkit-devel 
cd $INSTALL_FOLDER
git clone https://github.com/torch/distro.git torch --recursive
cd torch
TORCH_LUA_VERSION=$TORCH_VERSION sudo ./install.sh -b
source $HOME_USER/.bashrc
cd $THIS_FOLDER

# Install caffe
echo "Installing caffe library version $CAFFE_VERSION"
sudo yum install -y protobuf-compiler leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel
sudo yum install -y gflags-devel glog-devel lmdb-devel
cd ~/$INSTALL_FOLDER
git clone --branch v$CAFFE_VERSION https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
sed -i 's/BLAS := atlas/BLAS := open/' Makefile.config
sudo `which pip` install cython scikit-image h5py leveldb networkx python-gflags pillow
make all -j $(($(nproc) + 1))
make pycaffe -j $(($(nproc) + 1))
echo 'export CAFFE_ROOT=$(pwd)' >> $HOME_USER/.bashrc
echo 'export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH' >> $HOME_USER/.bashrc
source $HOME_USER/.bashrc
cd $THIS_FOLDER

# Install mxnet



