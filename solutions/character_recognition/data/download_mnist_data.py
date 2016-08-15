#Downloads the MNIST data

from deep_learning_tools_for_dsvm.solutions.utils.python_utils import *
import os
os.path.dirname(os.path.abspath(__file__))

url_train = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/mnist_train.csv'
url_test = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/mnist_test.csv'
print("Downloading file %s" % url_train)
download_file(url_train)
print("Downloading file %s" % url_test)
download_file(url_test)
