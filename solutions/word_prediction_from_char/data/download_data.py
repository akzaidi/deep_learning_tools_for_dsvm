
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file


url = 'https://hoaphumanoidstorage2.blob.core.windows.net/public/tiny_shakespeare'
download_file(url)