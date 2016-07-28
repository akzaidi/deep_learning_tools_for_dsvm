#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
THIS_FOLDER=$PWD
INSTALL_FOLDER=installer
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6
SKLEARN_VERSION=0.17.1
OPENBLAS_VERSION=0.2.18

# Create installation folder
mkdir ~/$INSTALL_FOLDER

#Install theano
echo "Installing theano library version $THEANO_VERSION"
sudo `which pip` install theano==$THEANO_VERSION

# Install keras
echo "Installing keras library version $KERAS_VERSION"
sudo `which pip` install keras==$KERAS_VERSION

# Install scikit-learn
echo "Installing scikit-learn library version $SKLEARN_VERSION"
sudo `which pip` install scikit-learn==$SKLEARN_VERSION

# Install openblas
echo "Installing open-blas version $OPENBLAS_VERSION"
cd ~/$INSTALL_FOLDER
git clone --branch v$OPENBLAS_VERSION https://github.com/xianyi/OpenBLAS/
make FC=gfortran -j $(($(nproc) + 1))
sudo make PREFIX=/usr/local install
cd $THIS_FOLDER





