# CUDA install in Ubuntu
# http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf

apt-get install build-essentials gfortran -y

INSTALL_FOLDER=$1
cd $INSTALL_FOLDER
mkdir CUDA
cd CUDA

#installation with deb
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
dpkg -i cuda*.deb
apt-get update
apt-get install cuda -y


