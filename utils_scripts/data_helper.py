import glob as glob
from skimage import io
import numpy as np

def get_images_as_matrix(data_path, file_type='tif'):
    image_files = glob.glob(data_path + '*' + file_type)

    assert len(image_files) != 0, "0 files found in the directory {}".format(data_path)

    images = []
    for filename in image_files:
        this_file_image = io.imread(filename)
        # tiffile sometimes changes the channel to the last dimension
        # if this_file_image.shape[-1] != 4 or this_file_image.shape[-1] != 25:
        #     this_file_image = np.transpose(this_file_image, (2,0,1))
        if this_file_image.shape[-1] == 4 :
            this_file_image = np.transpose(this_file_image, (2,0,1))

        images.append(this_file_image)

    return images
        

