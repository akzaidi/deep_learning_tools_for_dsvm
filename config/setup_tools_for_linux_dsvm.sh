#!/bin/bash
#
# This script installs several libraries for developing deep learning applications
#

# Script specifications, change the versions to fit preferences
INSTALL_FOLDER=$1/installer
SESSION_HOME=$2
THIS_FOLDER=$PWD
OPENBLAS_VERSION=0.2.18
THEANO_VERSION=0.8.2
KERAS_VERSION=1.0.6
SKLEARN_VERSION=0.17.1
CHAINER_VERSION=1.12.0
TORCH_VERSION=LUAJIT21
CAFFE_VERSION=rc3
MXNET_VERSION=20160531

# Create installation folder
echo "Initializing deep learning tools for dsvm script"
mkdir -p $INSTALL_FOLDER

# Install openblas
# FIXME: check if the version is the same, if not, then update
echo "Installing OpenBLAS version $OPENBLAS_VERSION"
OPENBLAS_FILE=/usr/local/include/openblas_config.h
if [ ! -e $OPENBLAS_FILE ] ; then
	cd $INSTALL_FOLDER
	git clone --branch v$OPENBLAS_VERSION https://github.com/xianyi/OpenBLAS/
	cd OpenBLAS
	make FC=gfortran -j $(nproc)
	make PREFIX=/usr/local install
	echo "export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" >> $SESSION_HOME/.bashrc
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
echo "Installing torch library version $TORCH_VERSION"
yum install -y cmake readline-devel ncurses-devel libjpeg-turbo-devel libpng-devel GraphicsMagick-devel fftw-devel sox-devel sox qt-devel qtwebkit-devel 
cd $INSTALL_FOLDER
git clone https://github.com/torch/distro.git torch --recursive
cd torch
TORCH_LUA_VERSION=$TORCH_VERSION ./install.sh -b
cd $THIS_FOLDER

# Install caffe
echo "Installing caffe library version $CAFFE_VERSION"
yum install -y protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel
yum install -y gflags-devel glog-devel lmdb-devel
cd $INSTALL_FOLDER
git clone --branch $CAFFE_VERSION https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
sed -i "s|# CPU_ONLY := 1|CPU_ONLY := 1|" Makefile.config
sed -i "s|BLAS := atlas|BLAS := open|" Makefile.config
sed -i "s|PYTHON_INCLUDE := /usr/include/python2.7 |# PYTHON_INCLUDE := /usr/include/python2.7 |" Makefile.config
sed -i "s|/usr/lib/python2.7/dist-packages/numpy/core/include|# /usr/lib/python2.7/dist-packages/numpy/core/include|"  Makefile.config
sed -i "s|# ANACONDA_HOME := \$(HOME)/anaconda|ANACONDA_HOME := /anaconda|" Makefile.config
sed -i "s|# PYTHON_INCLUDE := \$(ANACONDA_HOME)/include|PYTHON_INCLUDE := \$(ANACONDA_HOME)/include \$(ANACONDA_HOME)/include/python2.7 \$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include|" Makefile.config
sed -i "s|PYTHON_LIB := /usr/lib|# PYTHON_LIB := /usr/lib|" Makefile.config
sed -i "s|# PYTHON_LIB := \$(ANACONDA_HOME)/lib|PYTHON_LIB := \$(ANACONDA_HOME)/lib|" Makefile.config
`which pip` install cython scikit-image h5py leveldb networkx python-gflags pillow
make all -j $(nproc) 
make pycaffe -j $(nproc)
CAFFE_ROOT=$INSTALL_FOLDER/caffe
echo "export CAFFE_ROOT=$CAFFE_ROOT" >> $SESSION_HOME/.bashrc
PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
echo "export PYTHONPATH=$PYTHONPATH" >> $SESSION_HOME/.bashrc
cd $THIS_FOLDER

# Install mxnet
echo "Installing mxnet library version $MXNET_VERSION"
yum install -y libssh2-devel
cd $INSTALL_FOLDER
git clone --branch $MXNET_VERSION --recursive https://github.com/dmlc/mxnet.git
cd mxnet
cp make/config.mk .
sed -i "s|USE_BLAS = atlas|USE_BLAS = openblas|" config.mk
# sed -i "s|# TORCH_PATH = \$(HOME)/torch |# TORCH_PATH = $INSTALL_FOLDER/torch |" config.mk
# sed -i "s|# MXNET_PLUGINS += plugin/torch/torch.mk|MXNET_PLUGINS += plugin/torch/torch.mk|" config.mk
make -j $(nproc)
yum install python-setuptools
cd python
sed -i "s|'numpy',|# 'numpy',|" setup.py
python setup.py install
PYTHONPATH=$INSTALL_FOLDER/mxnet/python:$PYTHONPATH
echo "export PYTHONPATH=$PYTHONPATH" >> $SESSION_HOME/.bashrc
cd ..
Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
Rscript -e "install.packages(c('scales','knitr','mlbench','zoo','stringr','ggplot2','plyr','manipulate','colorspace','reshape2','digest','RColorBrewer'), dependencies = TRUE)"
cd ..
make rpkg

source $SESSION_HOME/.bashrc

cd $INSTALL_FOLDER/mxnet
Rscript  -e "LIB_PATH <- paste0(Sys.getenv('LD_LIBRARY_PATH'),':/usr/local/lib'); Sys.setenv(LD_LIBRARY_PATH=LIB_PATH); install.packages('mxnet_0.7.tar.gz')"

echo "Deep learning tools for dsvm script finished"

