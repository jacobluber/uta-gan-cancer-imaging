import tiffile as tif 
import torch
import os

def save_image( image_tensor, path):
    np_image = image_tensor[0].squeeze().cpu().float().numpy()
    tif.imwrite(path + '*.tif', np_image, compression='zlib', photometric='minisblack')

def save(target, gen, path, epoch, image=False, model=False):
    if image:
        path_real =  path + '/epoch_{}_'.format(epoch) + 'real_'
        path_fake = path + '/epoch_{}_'.format(epoch) + 'fake_'
        print(target.shape, target.dtype)
        save_image(target, path_real)
        save_image(gen, path_fake)

    elif model:
        path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D.pt')
        path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G.pt')
        torch.save(package['D_state_dict'], path_D)
        torch.save(package['G_state_dict'], path_G)