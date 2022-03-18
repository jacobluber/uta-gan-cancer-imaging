
'''
Filename: uta-gan-cancer-imaging/codex_data_module.py
Author: mxs2361

'''


import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from skimage import exposure, img_as_ubyte, io, transform
from utils_scripts.data_helper import get_images_as_matrix
from typing import Optional
import torch
import numpy as np

class CODEXDataModule(pl.LightningDataModule):

    def __init__(self, src_data_dir: str ='data/source/', tgt_data_dir: str='data/target/', data_format = 'tif', demo_data = False):
        super().__init__()
        self.src_data_dir = src_data_dir
        self.tgt_data_dir = tgt_data_dir
        self.data_format = data_format
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.demo_data = demo_data


    def prepare_data(self):
        if self.demo_data:
            self.src_images = torch.rand([2,25, 1024, 1024]).to(torch.float32)
            self.tgt_images = torch.rand([2,4, 1024, 1024]).to(torch.float32)
            self.images =[(src, tgt) for src, tgt in zip(self.src_images, self.tgt_images)] 
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        else:
            self.src_images = get_images_as_matrix(self.src_data_dir,self.data_format) 
            self.tgt_images = get_images_as_matrix(self.tgt_data_dir, self.data_format)
            self.images = [(src.astype(np.float32), tgt.astype(np.float32)) for src, tgt in zip(self.src_images, self.tgt_images)]

    def setup(self, stage: Optional[str] = None):
        # There are also data operations you might want to perform on every GPU. Use setup to do things like: transform
        self.dims = self.src_images[0].shape
    # serWarning: num_workers>0, persistent_workers=False, and strategy=ddp_spawn may result in data loading bottlenecks. Consider setting persistent_workers=True (this is a limitation of Python .spawn() and PyTorch)
    def train_dataloader(self):
        return DataLoader(self.images, batch_size = 2, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
    
    



def test():
    c = CODEXDataModule(src_data_dir = '/home/mxs2361/Dataset/codex_data/Data_scaled/train_A/', \
            tgt_data_dir='/home/mxs2361/Dataset/codex_data/Data_scaled/train_B/' )
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



