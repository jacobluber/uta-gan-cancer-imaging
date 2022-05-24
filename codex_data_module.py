
'''
Author: mxs2361

'''


from glob import glob
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from skimage import exposure, img_as_ubyte, io, transform
from utils_scripts.data_helper import get_images_as_matrix, get_image_as_matrix_with_metadata, get_src_target_split
from typing import Optional
import torch
import numpy as np
from create_data import TrainingFileCreation
import json

class CODEXDataModule(pl.LightningDataModule):

    def __init__(self, src_data_dir: str ='data/source/', tgt_data_dir: str='data/target/', raw_data_dir = None,\
    src_ch: int=25, tgt_ch: int=4, src_channel_ids: list= None, tgt_channel_ids: list=None, data_format = 'tif', tiles: bool= False, data_mode = "data_split", images = None):
        super().__init__()
        self.src_data_dir = src_data_dir
        self.tgt_data_dir = tgt_data_dir
        self.raw_data_dir = raw_data_dir
        self.data_format = data_format
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_mode = data_mode
        self.src_ch = src_ch
        self.tgt_ch = tgt_ch
        self.src_channel_ids = src_channel_ids
        self.tgt_channel_ids = tgt_channel_ids
        self.train_data_precentige = 0.8 
        self.images = images
        self.tiles = tiles




    def prepare_data(self):
        
        if self.data_mode == 'demo':
            self.src_images = torch.rand([20, self.src_ch, 1024, 1024]).to(torch.float32)
            self.tgt_images = torch.rand([20, self.tgt_ch, 1024, 1024]).to(torch.float32)
            self.images =[(src, tgt) for src, tgt in zip(self.src_images, self.tgt_images)] 
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        elif self.data_mode == 'data_split':
            self.src_images = get_images_as_matrix(self.src_data_dir, self.src_ch, self.data_format) 
            self.tgt_images = get_images_as_matrix(self.tgt_data_dir, self.tgt_ch, self.data_format)
            
            self.images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(self.src_images, self.tgt_images)]
        
        elif self.data_mode == 'raw_data_hubmap':
            import pandas as pd
            raw_file_dir = self.raw_data_dir
            df = pd.read_csv('/home/mxs2361/projects/hubmap_data_analysis/codex_meta_info.csv')

            images_29_channel = df[df['channel'] == 29] # Testing
            filenames = list(images_29_channel['filename'])

            print(len(filenames))
            print(filenames)
            #raw_data_scaled/HBM347.PSLC.425/reg1_stitched_expressions.ome.tif
            print(raw_file_dir)
            filepaths = [raw_file_dir + filename for filename in filenames]
            print(filepaths)

            filepaths = [raw_file_dir + '_'+filename.split('/')[-2] \
                + '_reg1_stitched_expressions.ome.tif' for filename in filenames] 

            t =  TrainingFileCreation(raw_filepaths = filepaths, rescale_shape = (1024, 1024), tiles=self.tiles, write_to_disk = False,\
                input_channel= self.src_ch, output_channel = self.tgt_ch, \
                    input_channel_ids = self.src_channel_ids, target_channel_ids=self.tgt_channel_ids,rescale_and_min_exposure = False,
            )
            self.src_images, self.tgt_images = t.create_data_from_raw_files()
            self.images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(self.src_images, self.tgt_images)]
        
        
        elif self.data_mode == 'raw_data':
            
            raw_file_dir = self.raw_data_dir
            
            #raw_data_scaled/HBM347.PSLC.425/reg1_stitched_expressions.ome.tif

            filepaths = glob(raw_file_dir + '/*.tif')
            print(filepaths)
            assert len(filepaths) > 0,f"No File Found, for the dir {raw_file_dir}"

            t =  TrainingFileCreation(raw_filepaths = filepaths, rescale_shape = (1024, 1024), tiles=self.tiles, write_to_disk = False,\
                input_channel= self.src_ch, output_channel = self.tgt_ch, \
                    input_channel_ids = self.src_channel_ids, target_channel_ids=self.tgt_channel_ids,rescale_and_min_exposure = True,
            )
            self.src_images, self.tgt_images = t.create_data_from_raw_files()
            self.images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(self.src_images, self.tgt_images)] 
        elif self.data_mode == 'image_given':
            self.src_images, self.tgt_images = get_src_target_split(self.images)
            self.images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(self.src_images, self.tgt_images)]
        else:
            print(f'Invalid data mode {self.data_mode}')



    def setup(self, stage: Optional[str] = None):
        # There are also data operations you might want to perform on every GPU. Use setup to do things like: transform
        self.dims = self.src_images[0].shape
    # serWarning: num_workers>0, persistent_workers=False, and strategy=ddp_spawn may result in data loading bottlenecks. Consider setting persistent_workers=True (this is a limitation of Python .spawn() and PyTorch)
    def train_dataloader(self):
        self.train_val_split = int(self.train_data_precentige * len(self.src_images))
        self.val_sample = int(0.9 * self.train_val_split)
        return DataLoader(self.images[:self.val_sample], batch_size = 32, num_workers=2, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.images[self.val_sample: self.train_val_split], batch_size = 32, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self, test_mode = False):
        if test_mode:
            #Return all the images
            return DataLoader(self.images, batch_size = 32, num_workers=4, pin_memory=True, persistent_workers=True) 
        return DataLoader(self.images[-self.train_val_split:], batch_size = 32, num_workers=4, pin_memory=True, persistent_workers=True) 
    
    

def test_mode_image_given():
    images = get_images_as_matrix(opt.raw_data_dir, channel_size = 29, file_type='tif')
    data = CODEXDataModule(images=images) 

    
    data.prepare_data()
    train_dataloader = data.train_dataloader()
    train_data = next(iter(train_dataloader))
    src_image, tgt_image = train_data

    print(len(train_data))
    print(type(train_data))
    print(src_image.shape, tgt_image.shape)

def test():
    raw_data_dir = '/home/mxs2361/Dataset/codex_data/raw_data_scaled/'

    #Test with the split data
    # c = CODEXDataModule(src_data_dir = '/home/mxs2361/Dataset/codex_data/Data_scaled/train_A/', \
    #         tgt_data_dir='/home/mxs2361/Dataset/codex_data/Data_scaled/train_B/' )
    #Test with raw_data
    with open('channel_ids.json') as fp:
        channel_ids = json.load(fp)
    c = CODEXDataModule(src_data_dir = '/home/mxs2361/Dataset/codex_data/Data_scaled/train_A/', \
            tgt_data_dir='/home/mxs2361/Dataset/codex_data/Data_scaled/train_B/', \
                src_ch=16, tgt_ch=13,
            src_channel_ids= channel_ids['source_channel_ids'],
            tgt_channel_ids= channel_ids['target_channel_ids'],
            raw_data_dir=raw_data_dir, data_mode='raw_data' )
    c.prepare_data()
    train_batch = c.train_dataloader()
    train_data = next(iter(train_batch))
    src_image, tgt_image = train_data

    print(len(train_data))
    print(type(train_data))
    print(src_image.shape, tgt_image.shape)
    # src_img, tgt_img = train_data
    # print('src image shape ', src_img[0].shape)
    # print('target image shape ', tgt_img[0].shape)


if __name__ == '__main__':
    test()



