#!/bin/bash

#Downloads the MNIST data

wget https://hoaphumanoidstorage2.blob.core.windows.net/public/Inception.zip
unzip Inception.zip
cd Inception/Inception
cp * ../../
cd ../../
rm -rf Inception