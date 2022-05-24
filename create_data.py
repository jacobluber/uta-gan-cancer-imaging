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
        input_channel: int = 25, output_channel: int = 4, input_channel_ids = None, target_channel_ids = None, \
        rescale_and_min_exposure: bool = True, tiles: bool = False,\
        tile_size: int = 512, write_to_disk: bool = False, write_data_dir: str = None):
        
        
        self.raw_filepaths = raw_filepaths
        self.rescale_shape = rescale_shape
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.rescale_and_min_exposure = rescale_and_min_exposure
        self.tiles = tiles
        self.tile_size = tile_size
        self.write_to_disk = write_to_disk
        self.write_data_dir = write_data_dir

        self.input_channel_ids = input_channel_ids
        self.target_channel_ids = target_channel_ids
        



        if write_to_disk:
            pathlib.Path(write_data_dir + '/train_A/').mkdir(parents=True, exist_ok=True)
            pathlib.Path(write_data_dir + '/train_B/').mkdir(parents=True, exist_ok=True)
    
    def get_selected_channels(self, this_image, input_channels, target_channels = None):
        #For consecutive channels
        if isinstance(input_channels, int):
            src_image = this_image[:input_channels, :, :] # Select first n channels as condition image
            tgt_image = this_image[input_channels:, :, :] # Select last 29 - n channels as target image
            return src_image, tgt_image
        
        src_image = []
        tgt_image = []
        for channel_id in input_channels:
            src_image.append(this_image[channel_id, :, :])
        
        for channel_id in target_channels:
            tgt_image.append(this_image[channel_id, :, :])

        return np.asarray(src_image), np.asarray(tgt_image)


    
    def create_data_from_raw_files(self):
        src_images = []
        tgt_images = []
        count = 0
        for filepath in self.raw_filepaths:
            this_image = io.imread(filepath)

            # src_image = this_image[:self.input_channel, :, :] # Select first n channels as condition image
            # tgt_image = this_image[self.input_channel:, :, :] # Select last 29 - n channels as target image

            # input_channels = [i for i in range(self.input_channel)]
            # target_channels = [i for i in range(self.input_channel, this_image.shape[0])]
            print(self.input_channel_ids, self.target_channel_ids)
            src_image, tgt_image = self.get_selected_channels(this_image, input_channels=self.input_channel_ids, target_channels=self.target_channel_ids)

            if not self.tiles and self.rescale_and_min_exposure:
                if src_image.shape[1] <= 1024:
                    print("image is already scaled, scalling skipped")
                else:
                    src_image = TrainingFileCreation.rescale_image(src_image, self.rescale_shape)
                    tgt_image = TrainingFileCreation.rescale_image(tgt_image, self.rescale_shape)

            assert src_image.shape[0] == self.input_channel, f"Source image dimension mismatch, src_image channel {src_image.shape[0]}, required input channel {self.input_channel}"
            assert tgt_image.shape[0] == self.output_channel, "Target image dimension mismatch"

            if self.write_to_disk:
                img_file_name = filepath.split('/')[-1]
                slide_id = filepath.split('/')[-2]
                img_file_name = slide_id + '_' + img_file_name
                if self.tiles:
                    continue
                    # Create tiles for source image
                    # self.save_tiles(img=src_image, img_file_name=img_file_name, src=True)
                    # self.save_tiles(img=tgt_image, img_file_name=img_file_name, src=False)

                else:
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_A/_' + img_file_name), src_image) # Writing condition image
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_B/_' + img_file_name), tgt_image) # Writing target image
            
            elif self.tiles:
                img_file_name = filepath.split('/')[-1]
                slide_id = filepath.split('/')[-2]
                img_file_name = slide_id + '_' + img_file_name
                tiles_for_this_src_image = self.get_tiles(img=src_image, img_file_name=img_file_name, src=True)
                tiles_for_this_tgt_image = self.get_tiles(img=tgt_image, img_file_name=img_file_name, src=False)
                print(len(tiles_for_this_src_image), type(tiles_for_this_src_image))
                src_images.extend(tiles_for_this_src_image)
                tgt_images.extend(tiles_for_this_tgt_image)
                

            
            else:
                src_images.append(src_image)
                tgt_images.append(tgt_image)
            print('Count ', count )
            
            count += 1
        
        if not self.write_to_disk:
            print('Prepared image shape ', np.asarray(src_images[0]).shape)
            return src_images, tgt_images
    
    
    def get_tiles(self, img, img_file_name,src):
        this_image_tiles = []
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
                if self.write_to_disk:
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_A/_tile_{}'.format(tile_id) + img_file_name), tile) 
                this_image_tiles.append(tile)
        else:
           for tile_id, tile in tiler.iterate(img):
               tile = exposure.rescale_intensity(tile, out_range=(0,255))
               this_image_tiles.append(tile)
               if self.write_to_disk:
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/train_B/_tile_{}'.format(tile_id) + img_file_name), tile) 
                

        return this_image_tiles

    def scale_data_and_save(self):
        images = [] 
        count = 0
        for filepath in self.raw_filepaths:
            this_image = io.imread(filepath)
            
            if not self.tiles:
                this_image = TrainingFileCreation.rescale_image(this_image, self.rescale_shape)


            if self.write_to_disk:
                img_file_name = filepath.split('/')[-2] + '_' + filepath.split('/')[-1]

                
                if self.tiles:
                    slide_id = filepath.split('/')[-2]
                    img_file_name = slide_id + '_' + img_file_name
                    # Create tiles for source image
                    self.save_tiles(img=this_image, img_file_name=img_file_name, src=True)
                else:
                    TrainingFileCreation.write_file(str(self.write_data_dir + '/_' + img_file_name), this_image) # Writing condition image
            
            else:
                images.append(this_image)
            print('Count ', count )
            
            count += 1
        
        if not self.write_to_disk:
            return images 
    
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


def convert_to_scaled_files(files, new_dir, shape=(1024,1024)):
    for filename in files:
        print(filename)
        this_image = io.imread(filename)
        this_image = TrainingFileCreation.rescale_image(this_image, shape) 
        new_file_name = filename.split('/')[-1]
        new_path = new_dir + 'scaled_' + new_file_name
        TrainingFileCreation.write_file(new_path, this_image)
def add_random_channel(files, new_dir, shape=(1024,1024)):
    for filename in files:
        print(filename)
        this_image = io.imread(filename)
        shape = (this_image.shape[1],this_image.shape[2])
        print(shape)
        this_image = this_image.tolist()
        random_image = np.random.randint(1, 255, size=shape)
        this_image.append(random_image)
        this_image = np.asarray(this_image)
        print(this_image.shape)
        new_file_name = filename.split('/')[-1]
        new_path = new_dir + 'dimension_added_' + new_file_name
        TrainingFileCreation.write_file(new_path, this_image)

if __name__ == '__main__':
    import pandas as pd
    
    raw_file_dir = '/home/mxs2361/Dataset/codex_data/nih_tiff_images/'

    files = glob.glob(raw_file_dir + '*.tif')

    # print(files)
    # convert_to_scaled_files(files, '/home/mxs2361/Dataset/codex_data/nih_tiff_images_scaled/')
    add_random_channel(files, '/home/mxs2361/Dataset/codex_data/nih_tiff_images_random_channel_added/')
    # df = pd.read_csv('/home/mxs2361/projects/hubmap_data_analysis/codex_meta_info.csv')


    # images_29_channel = df[df['channel'] == 29] 

    # filenames = list(images_29_channel['filename'])

    # filepaths = [raw_file_dir + filename for filename in filenames]

    # # t =  TrainingFileCreation(raw_filepaths = filepaths, rescale_shape = (1024, 1024), tiles=False, write_to_disk = True,\
    # #  write_data_dir  = '/home/mxs2361/Dataset/codex_data/raw_data_scaled')
    # # # t.create_data_from_raw_files()
    # # t.scale_data_and_save()
    # filepaths = filepaths[:2]

    # t =  TrainingFileCreation(raw_filepaths = filepaths, rescale_shape = (1024, 1024), tiles=True, write_to_disk = False,\
    #             input_channel= 19, output_channel = 10,\
    #                 input_channel_ids=[19, 17, 7, 23, 11, 27, 10, 13, 22, 15, 26, 18, 24, 8, 25, 5],\
    #                 target_channel_ids=[0, 1, 2, 3, 4, 6, 9, 12, 14, 16, 20, 21, 28], rescale_and_min_exposure = False,
    #         )
    # src_images, tgt_images = t.create_data_from_raw_files()
    # for image in src_images:
    #     print(len(image), type(image))
    #     print(image.shape)
    # images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(src_images, tgt_images)]







