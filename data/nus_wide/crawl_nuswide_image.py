
import re
from urllib.request import urlretrieve
import os
# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from pathlib import Path

nuswide = "./NUS-WIDE-urls.txt" #the location of your nus-wide-urls.txt
image_dir = "./images/" # path of dataset you want to download in

f = open(nuswide, 'r')
url_lines = list(f.readlines())[1:]
f.close()

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def getsource(line):
    line = [item for item in line.split(" ") if item]
    img_url = line[2]
    img_name = image_dir + line[1] + ".jpg"
    print(img_name)

    if not os.path.exists(img_name):
        try:
            urlretrieve(img_url, img_name)
        except Exception:
            pass
    else:
        pass

pool = Pool(10)
results=pool.map(getsource, url_lines)
pool.close()
pool.join()
print("Done")
