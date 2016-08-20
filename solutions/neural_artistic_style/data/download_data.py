#Downloads the data
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file

url = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/neural_stile_vgg19.params'
download_file(url)

