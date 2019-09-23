from PIL import Image
import PIL.ImageOps as imops
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from pystruts import ImageImporter

#take edt
#identify pixels on the edge
#turn them off

#for expansion
#invert image
#identify pixels on edge
#turn them on

class Data(object):
    def __init__(self):
        self.data = {}
    
in_dir = '/data/downsample-512/'
imdata = Data().data
    
image = ImageImporter(imdata, in_dir).import_image('0-lx')
print(image)