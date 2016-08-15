#Downloads the ImageNet data

from deep_learning_tools_for_dsvm.solutions.utils.python_utils import *
import os
import zipfile
os.path.dirname(os.path.abspath(__file__))


url = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/Inception.zip'
print("Downloading file %s" % url)
download_file(url)
local_filename = url.split('/')[-1]
zfile = zipfile.ZipFile(local_filename)
for name in zfile.namelist():
    (dirname, filename) = os.path.split(name)
    print "Decompressing " + filename + " on " + dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    zfile.extract(name, dirname)