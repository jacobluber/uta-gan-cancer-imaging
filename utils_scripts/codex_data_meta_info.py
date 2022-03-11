

#This file is for concatenating source and target image for pix2pix implementation

#necessary imports

import glob
import numpy as np
from skimage import io as ski_io
import plotly.express as px
import matplotlib.pyplot as plt
from utils import get_channel_names
import tifffile
import pandas as pd
import concurrent
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, const='/home/mxs2361/Downloads/*/')
args = parser.parse_args()
#data dir
data_dir = args.data_dir #expected each image file under each sample name
files = glob.glob(data_dir + '*.tif')

# files = files[:10]

def process_file(filename):
    this_file_metadata = {}
    img = ski_io.imread(filename)
    filename_strip =  filename.split('/')[-2] +'/' +filename.split('/')[-1]
    
    this_file_metadata['shape'] = img.shape
    this_file_metadata['channel'] = img.shape[0]
    this_file_metadata['filename'] = filename_strip 
    return this_file_metadata

def process_files_executors(files):
  meta_info_dicts = [] 
  
  start_time = time.time()
  
  executor = concurrent.futures.ProcessPoolExecutor(len(files))
  futures = [executor.submit(process_file, filename) for filename in files]
  concurrent.futures.wait(futures)
  
  end_time = time.time()
  
  for _ in concurrent.futures.as_completed(futures):
    meta_info_dicts.append(_.result())
  
  end_time = time.time()
  print('elapsed time for threadpool ', end_time - start_time)
  return meta_info_dicts


def process_files(files):
  meta_info_dicts = []
  start_time = time.time()
  
  for filename in files:
    res = process_file(filename)
    meta_info_dicts.append(res)
  
  end_time = time.time()

  print('elapsed time for loop ', end_time - start_time)

if __name__ == '__main__':
  meta_info_dicts = process_files_executors(files)
  # process_files(files)
  df = pd.DataFrame(meta_info_dicts)  

  df.to_csv('codex_meta_info.csv', index = False)


