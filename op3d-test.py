import pydirectory as pyd
# import open3d as otd
import numpy as np
from stl import mesh
import numpy
import scipy as sp
import PIL.Image as Image
import matplotlib.pyplt as plt
from mpl_toolkits.mplot3d import Axes3D
import os

in_dir = "/data/sample1/25/uct/tiff/"

input_dir = pyd.Directory(in_dir).InputDIR()


class Voxelize:
    def __init__(self, directory):
        self.directory = directory

    def toNumpy(self):
        res = np.asarray(Image.open(self.directory+os.listdir(self.directory)[0])).shape
        arr = np.empty([res[0], res[1], len(os.listdir(self.directory))])
        # print(arr.shape)
        for i,j in enumerate(os.listdir(self.directory)):
            arr[:,:,i] = np.asarray(Image.open(self.directory+j))
            # print(im.shape)
            # print(arr[i].shape)
            # arr[:,:,i] = im
        # print(arr)
        return arr

    def saveSTL(self, filename):
        pass

    def getBoundingBox(self):
        pass


a = Voxelize(input_dir).toNumpy()
figure = plt.figure()
axes = Axes3d(figure)
mesharr = mesh.Mesh.