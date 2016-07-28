#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#
# Script specifications, change
INSTALL_FOLDER=installer
KERAS_VERSION=1.0.6

# Create installation folder
mkdir ~/$INSTALL_FOLDER

# Install keras
echo "Installing keras library version $KERAS_VERSION"
pip install keras==$KERAS_VERSION


