import glob as glob
from skimage import io as skiio
import numpy as np
from tifffile import TiffFile
from xml.etree import ElementTree
import xml
import io

def get_images_as_matrix(data_path, channel_size, file_type='tif'):
    image_files = glob.glob(data_path + '*' + file_type)

    # image_files = image_files[:10]

    assert len(image_files) != 0, f"0 files found in the directory {data_path}"

    images = []
    for filename in image_files:
        this_file_image = skiio.imread(filename)
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

def get_src_target_split(images, input_channel, output_channel):
    src_images = []
    tgt_images = []
    for this_image in images:
        src_image = this_image[input_channel, :, :] # Select first n channels as condition image
        tgt_image = this_image[input_channel:, :, :] # Select last 29 - n channels as target image
        src_images.append(src_image)
        tgt_images.append(tgt_image)
    return src_images, tgt_images

def read_tiff_image_and_meta(filename):
    with TiffFile(filename) as tif:
        
        images = tif.asarray()
        
        omexml_string = tif.pages[0].description

        root = xml.etree.ElementTree.parse(io.StringIO(omexml_string))
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
        channel_names = [c.attrib['Name'] for c in channels]

        return images, channel_names

def get_image_as_matrix_with_metadata(data_path, channel_size, file_type='tif'):
    image_files = glob.glob(data_path + '*' + file_type)

    # image_files = image_files[:10]

    assert len(image_files) != 0, f"0 files found in the directory {data_path}"

    images = []
    channels = []
    for filename in image_files:
        this_file_image, channel_names = read_tiff_image_and_meta(filename)
        if channel_size == 1:
            print(this_file_image.shape)
            this_file_image = this_file_image[..., np.newaxis]
            print(this_file_image.shape)
        if this_file_image.shape[-1] == channel_size :
            
            this_file_image = np.transpose(this_file_image, (2,0,1))
            
            
        images.append(this_file_image)
        channels.append(channel_names)

    return images, channels