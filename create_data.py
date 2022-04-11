# This file is for creating data from the raw images

from skimage import exposure, img_as_ubyte, io, transform
import numpy as np
import tifffile as tif 
import cv2 
import glob
from skimage import exposure, img_as_ubyte, io, transform
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import scipy.misc
import imageio
from multiprocessing import Pool
import pathlib
from tiler import Tiler, Merger



class TrainingFileCreation():
    """This file reads raw files and create source and target images split by channels (dest: write_data_dir/trainA and write_data_dir/trainB)
    Raw file directory is expected to be in the format of raw_file_dir/hubmap_id/*.tif
    """
    def __init__(self,  raw_filepaths: str, rescale_shape: tuple,\
        input_channel: int = 25, output_channel: int = 4, tiles: bool = False,\
             tile_size: int = 512, write_to_disk: bool = True, write_data_dir: str = None):
        
        
        self.raw_filepaths = raw_filepaths
        self.rescale_shape = rescale_shape
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.tiles = tiles
        self.tile_size = tile_size
        self.write_to_disk = write_to_disk
        self.write_data_dir = write_data_dir
        



        if write_to_disk:
            pathlib.Path(write_data_dir + '/train_A/').mkdir(parents=True, exist_ok=True)
            pathlib.Path(write_data_dir + '/train_B/').mkdir(parents=True, exist_ok=True)
    
    def create_data_from_raw_files(self):
        src_images = []
        tgt_images = []
        count = 0
        for filepath in self.raw_filepaths:
            this_image = io.imread(filepath)

            src_image = this_image[:self.input_channel, :, :] # Select first n channels as condition image
            tgt_image = this_image[self.input_channel:, :, :] # Select last 29 - n channels as target image

            if not self.tiles:
                src_image = TrainingFileCreation.rescale_image(src_image, self.rescale_shape)
                tgt_image = TrainingFileCreation.rescale_image(tgt_image, self.rescale_shape)

            assert src_image.shape[0] == self.input_channel, "Source image dimension mismatch"
            assert tgt_image.shape[0] == self.output_channel, "Target image dimension mismatch"

            if self.write_to_disk:
                img_file_name = filepath.split('/')[-1]
                slide_id = filepath.split('/')[-2]
                img_file_name = slide_id + '_' + img_file_name
                if self.tiles:
                    # Create tiles for source image
                    self.save_tiles(img=src_image, img_file_name=img_file_name, src=True)
                    self.save_tiles(img=tgt_image, img_file_name=img_file_name, src=False)

                else:
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_A/_' + img_file_name), src_image) # Writing condition image
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_B/_' + img_file_name), tgt_image) # Writing target image
            
            else:
                src_images.append(src_image)
                tgt_images.append(tgt_image)
            print('Count ', count )
            
            count += 1
        
        if not self.write_to_disk:
            return src_images, tgt_images
    
    
    def save_tiles(self, img, img_file_name,src):
        if src:
            channel = self.input_channel
        else:
            channel = self.output_channel
        tiler = Tiler(data_shape=img.shape,
                    tile_shape=(channel, self.tile_size, self.tile_size),
                    channel_dimension=0)
        if src:
            for tile_id, tile in tiler.iterate(img):
                tile = exposure.rescale_intensity(tile, out_range=(0,255))
                TrainingFileCreation.write_file(str(self.write_data_dir + '/train_A/_tile_{}'.format(tile_id) + img_file_name), tile) 
        else:
           for tile_id, tile in tiler.iterate(img):
               tile = exposure.rescale_intensity(tile, out_range=(0,255))
               TrainingFileCreation.write_file(str(self.write_data_dir + '/train_B/_tile_{}'.format(tile_id) + img_file_name), tile) 
    
    @staticmethod
    def rescale_image(image, shape):
        np_list_for_channels = []  
        for channel_id in range(image.shape[0]):
            this_channel_data = image[channel_id, :, :]
            this_channel_rescaled = transform.resize(this_channel_data, shape).astype(np.float32)
            # print('updated scale ', this_channel_rescale.shape)
            # this_channel_8bit = img_as_ubyte(exposure.rescale_intensity(this_channel_rescale))

            this_channel_rescaled = exposure.rescale_intensity(this_channel_rescaled, out_range=(0,255))
            np_list_for_channels.append(this_channel_rescaled)
        np_tuples_for_channels = tuple(np_list_for_channels)
        scaled_image = np.stack(np_tuples_for_channels)
        return scaled_image

    @staticmethod
    def write_file(path, img):
        tif.imwrite(path, img)

if __name__ == '__main__':
    import pandas as pd
    
    raw_file_dir = '/home/mxs2361/Dataset/codex_data/raw_data/'

    df = pd.read_csv('/home/mxs2361/projects/hubmap_data_analysis/codex_meta_info.csv')


    images_29_channel = df[df['channel'] == 29] 

    filenames = list(images_29_channel['filename'])

    print(len(filenames))
    filepaths = [raw_file_dir + filename for filename in filenames]

    t =  TrainingFileCreation(raw_filepaths = filepaths, rescale_shape = (1024, 1024), tiles=False, write_to_disk = True,\
     write_data_dir  = '/home/mxs2361/Dataset/codex_data/Data_scaled_20_9')
    t.create_data_from_raw_files()







