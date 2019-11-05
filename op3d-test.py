import pydirectory as pyd
import open3d as otd
import pymesh
import numpy as np
# import stl
import numpy
import scipy as sp
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pyassimp
from pyntcloud import PyntCloud
import pandas as pd

in_dir = "/data/sample1/25/uct/tiff/" #'/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

input_dir = pyd.Directory(in_dir).InputDIR()

# model = pyassimp.load(input_dir)
# print(len(model.meshes))

class Voxelize:
    def __init__(self, directory):
        self.directory = directory

    def toNumpy(self):
        res = np.asarray(Image.open(self.directory+os.listdir(self.directory)[0])).shape
        arr = np.empty([res[0], res[1], len(os.listdir(self.directory))])
        # print(arr.shape)
        for i,j in enumerate(os.listdir(self.directory)):
            arr[:,:,i] = np.asarray(Image.open(self.directory+j), dtype=bool)
            # print(im.shape)
            # print(arr[i].shape)
            # arr[:,:,i] = im
        # print(arr)
        return arr


a = Voxelize(input_dir).toNumpy()

points = pd.DataFrame(a)
print(points)

z,x,y = a.nonzero()
# b = a.toNumpy()
# print(b.shape)
# c = a.NumpytoVtk(b)
# print(c)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.voxels(a, edgecolor='k')
# plt.show()