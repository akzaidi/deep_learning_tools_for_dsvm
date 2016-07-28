#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
INSTALL_FOLDER=installer
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6

# Create installation folder
mkdir ~/$INSTALL_FOLDER

#Install theano
echo "Installing theano library version $THEANO_VERSION"
sudo `which pip` install theano==$THEANO_VERSION

# Install keras
echo "Installing keras library version $KERAS_VERSION"
sudo `which pip` install keras==$KERAS_VERSION


