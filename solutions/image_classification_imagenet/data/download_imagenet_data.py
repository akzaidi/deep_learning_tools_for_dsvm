#Downloads the ImageNet data
import zipfile
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file


url = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/Inception.zip'
print("Downloading file %s" % url)
download_file(url)
local_filename = url.split('/')[-1]
zfile = zipfile.ZipFile(local_filename)
for name in zfile.namelist():
    zfile.extract(name, current_path)