from PIL import Image
import PIL.ImageOps as imops
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from pystruts import ImageImporter, Filters
import matplotlib.pyplot as plt
import math
import random
import pydirectory as pyd

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

imdata = Data().data

in_dir = '/data/final_images/'
out_dir = pyd.Directory(in_dir, '/data/deg_test/').OutputDIR()
image = ImageImporter(imdata, in_dir).import_image('0-lx')

def iter(n):
    data_copy = imdata['0-lx']['raw_data']

    i = 0
    while i < n:
        i += 1
        dt = ndimage.distance_transform_edt(data_copy)
        edge = np.where(dt >= 2)

        randlist_size = random.uniform(0.3, 0.7)
        randlist = np.random.randint(0, len(edge[0]), math.floor(randlist_size*len(edge[0])))
        col1 = edge[0]
        col2 = edge[1]

        col1rand = [x for x in col1[randlist]]
        col2rand = [x for x in col2[randlist]]

        data_copy = pixel_off(data_copy, col1rand, col2rand)

        plt.imsave(out_dir+'0-lx'+'-'+str(i)+'.tif', data_copy, cmap='binary')


def pixel_off(image, col1, col2):
    for i in col1:
        for j in col2:
            image[i, j] == 0
    return image

iter(5)


# def iterate(image):
#     data_copy = data
#     dt = ndimage.distance_transform_edt(data_copy)
#     edge = np.where(dt == 1)

#     randlist_size = random.uniform(0.3, 0.7)
#     randlist = np.random.randint(0, len(edge[0]), math.floor(randlist_size*len(edge[0])))
#     col1 = edge[0]
#     col2 = edge[1]

#     col1rand = [x for x in col1[randlist]]
#     col2rand = [x for x in col2[randlist]]

#     for j in col1rand:
#         for k in col2rand:
#             data_copy[j, k] == 0

#     data_deg = data_copy
    
#     plt.imsave(out_dir+'0-lx'+'_'+str(i)+'.tif', data_deg, cmap='binary')

#     return data_deg


# for i in np.arange(10):
#     dt = ndimage.distance_transform_edt(data_copy)
#     edge = np.where(dt == 1)
    
#     randlist_size = random.uniform(0.3, 0.7)
#     randlist = np.random.randint(0, len(edge[0]), math.floor(randlist_size*len(edge[0])))
#     col1 = edge[0]
#     col2 = edge[1]

#     col1rand = [i for i in col1[randlist]]
#     col2rand = [i for i in col2[randlist]]

#     for i in col1rand:
#         for j in col2rand:
#             data_copy[i, j] == 0

#     data_copy = data_copy
    


# dt = ndimage.distance_transform_edt(imdata['0-lx']['raw_data'])

# # plt.imshow(dt)
# # plt.show()

# one_loc = np.where(dt == 1)
# # one_loc = zip(one_loc[0], one_loc[1])
# print(len(one_loc[0]))

# randlist_size = random.uniform(0.3, 0.7)
# print(randlist_size)

# randlist = np.random.randint(0, len(one_loc[0]), math.floor(randlist_size*len(one_loc[0])))
# print(len(randlist))
# print(min(randlist), max(randlist))
# print(randlist)

# col1 = one_loc[0]
# col2 = one_loc[1]

# col1_rand = [i for i in col1[randlist]]
# col2_rand = [i for i in col2[randlist]]

# data_copy = imdata['0-lx']['raw_data']

# for i in col1_rand:
#     for j in col2_rand:
#         data_copy[i, j] == 0

# plt.imshow(data_copy)
# plt.show()



# plt.imshow(imdata['0-lx']['raw_data'])
# plt.show()