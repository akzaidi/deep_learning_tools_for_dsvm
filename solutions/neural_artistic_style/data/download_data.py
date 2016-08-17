#Downloads the data
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file


url1 = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/starry_night.jpg'
url2 = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/satya.jpg'
url3 = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/bill-gates-desk.jpg'
url4 = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/neural_stile_vgg19.params'
download_file(url1)
download_file(url2)
download_file(url3)
download_file(url4)

