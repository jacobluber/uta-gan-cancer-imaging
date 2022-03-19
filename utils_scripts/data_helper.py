import glob as glob
from skimage import io
import numpy as np

def get_images_as_matrix(data_path, channel_size, file_type='tif'):
    image_files = glob.glob(data_path + '*' + file_type)

    # image_files = image_files[:10]

    assert len(image_files) != 0, f"0 files found in the directory {data_path}"

    images = []
    for filename in image_files:
        this_file_image = io.imread(filename)
        # tiffile sometimes changes the channel to the last dimension
        # if this_file_image.shape[-1] != 4 or this_file_image.shape[-1] != 25:
        #     this_file_image = np.transpose(this_file_image, (2,0,1))
        if channel_size == 1:
            print(this_file_image.shape)
            this_file_image = this_file_image[..., np.newaxis]
            print(this_file_image.shape)
        if this_file_image.shape[-1] == channel_size :
            
            this_file_image = np.transpose(this_file_image, (2,0,1))
            
            
        images.append(this_file_image)

    return images
        

