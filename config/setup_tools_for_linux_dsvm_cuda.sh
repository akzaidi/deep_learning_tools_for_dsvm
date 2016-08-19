# CUDA install in Ubuntu

INSTALL_FOLDER=$1
cd $INSTALL_FOLDER
mkdir CUDA
cd CUDA
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb
dpkg -i cuda-*.deb
apt-get update
apt-get install cuda