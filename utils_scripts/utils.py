import tiffile as tif 
import torch
import os

def save_image( image_tensor, path, save_all = False):
    if save_all:
        print('Total image ', image_tensor.shape[0])
        for i in range(image_tensor.shape[0]):
            np_image = image_tensor[i].squeeze().cpu().float().numpy()
            print(path)
            tif.imwrite(path + f'{i}*.tif', np_image, compression='zlib', photometric='minisblack') 
            print('file written')
    else:
        np_image = image_tensor[0].squeeze().cpu().float().numpy()
        tif.imwrite(path + '*.tif', np_image, compression='zlib', photometric='minisblack')

def save(target, gen, path, epoch, image=False, model=False, train_status = 'train'):
    if image:
        path_real =  path + f'/{train_status}_epoch_{epoch}_real'
        path_fake = path + f'/{train_status}_epoch_{epoch}_fake'
        print(target.shape, target.dtype)
        save_image(target, path_real)
        save_image(gen, path_fake)

    elif model:
        path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D.pt')
        path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G.pt')
        torch.save(package['D_state_dict'], path_D)
        torch.save(package['G_state_dict'], path_G)