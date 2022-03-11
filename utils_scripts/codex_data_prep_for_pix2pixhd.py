


import tifffile
from PIL import Image
import os
import pandas as pd
from skimage import io

root_data_dir = '/home/mxs2361/Downloads/'
split_data_dir_A = 'Data/train_A/'
split_data_dir_B = 'Data/train_B/'
df = pd.read_csv('codex_meta_info.csv')

images_29_channel = df[df['channel'] == 29]

files = list(images_29_channel['filename'])

print(len(files))

for filename in files:
  src_img_path = root_data_dir + filename
  img = io.imread(src_img_path)
  file_dir_A = split_data_dir_A + filename.split('/')[0] + '_'+ '.tif'
  file_dir_B = split_data_dir_B + filename.split('/')[0] + '_' + '.tif'
  tifffile.imsave(file_dir_A, img[:25,:,:])
  tifffile.imsave(file_dir_B, img[25:, :, :])

  



