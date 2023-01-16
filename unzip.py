import zipfile
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import time

from utils import *

settings = get_settings()
globals().update(settings)

from_dir = path_join(route, 'download')
des = path_join(route, 'unzip')

mkdir(des)

def fast_unzip(zip_path, out_path):
    print(zip_path)
    try:
        start = time.time()
        with ZipFile(zip_path) as handle:
            with ThreadPoolExecutor(2) as exe:
                _ = [exe.submit(handle.extract, m, out_path) for m in handle.namelist()]
    except:
        pass
    finally:
        print('Unzip', zip_path, 'Time:', time.time() - start)

filename = 'data'
zip_path = path_join(from_dir, filename + '.zip')
fast_unzip(zip_path, des)