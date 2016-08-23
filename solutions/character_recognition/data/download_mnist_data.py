#Downloads the MNIST data
import sys
import os
import glob
import gzip
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file

# MNIST data in csv
url_root = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/'
url_train = url_root + 'mnist_train.csv'
url_test = url_root + 'mnist_test.csv'
print("Downloading file %s" % url_train)
download_file(url_train)
print("Downloading file %s" % url_test)
download_file(url_test)

#MNIST data in ubyte
url_ubyte_train_im = url_root + 'train-images-idx3-ubyte.gz'
url_ubyte_train_lb = url_root + 'train-labels-idx1-ubyte.gz'
url_ubyte_test_im = url_root + 't10k-images-idx3-ubyte.gz'
url_ubyte_test_lb = url_root + 't10k-labels-idx1-ubyte.gz'
print("Downloading file %s" % url_ubyte_train_im)
download_file(url_ubyte_train_im)
print("Downloading file %s" % url_ubyte_train_lb)
download_file(url_ubyte_train_lb)
print("Downloading file %s" % url_ubyte_test_im)
download_file(url_ubyte_test_im)
print("Downloading file %s" % url_ubyte_test_lb)
download_file(url_ubyte_test_lb)
for gzip_file in glob.glob('*.gz'):
	print ("Extracting %s" % gzip_file)
	file = os.path.splitext(gzip_file)[0]
	if not os.path.isfile(file):
		with gzip.open(gzip_file, 'rb') as in_file:
			s = in_file.read()
		with open(file, 'w') as f:
			f.write(s)
	else:
		print ("File %s already exists" % file)