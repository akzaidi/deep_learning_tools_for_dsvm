#!/bin/bash
#
# This script installs a group of deep learning libraries and tools in a DSVM
#

# Prepare environment
for i in "$@"
do
case $i in
    -f=*|--install-folder=*)
    INSTALL_FOLDER="${i#*=}"
    shift # past argument=value
    ;;
esac
done
SESSION_USER=$USER
SESSION_HOME=$HOME
if [ -z "$INSTALL_FOLDER" ]; then
        INSTALL_FOLDER=$PWD
fi
echo "INSTALL FOLDER  = $INSTALL_FOLDER"
echo "USER = $SESSION_USER"
echo "HOME PATH = $SESSION_HOME"

# Execute script that installs many deep learning libraries
sudo config/setup_tools_for_linux_dsvm.sh $INSTALL_FOLDER $SESSION_HOME

# Install mxnet
#Rscript -e "LIB_PATH <- paste0(Sys.getenv('LD_LIBRARY_PATH'),'/usr/local/lib'); Sys.setenv(LD_LIBRARY_PATH=LIB_PATH)"
cd $INSTALL_FOLDER/mxnet
R CMD INSTALL mxnet_0.7.tar.gz


