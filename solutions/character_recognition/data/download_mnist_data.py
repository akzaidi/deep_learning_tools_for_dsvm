#Downloads the MNIST data
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file

url_train = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/mnist_train.csv'
url_test = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/mnist_test.csv'
print("Downloading file %s" % url_train)
download_file(url_train)
print("Downloading file %s" % url_test)
download_file(url_test)
