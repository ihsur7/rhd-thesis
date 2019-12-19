import pydirectory as pyd
import open3d as otd
import numpy as np
# import stl
import numpy
import scipy as sp
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

in_dir = "/data/sample1/25/uct/tiff/" #'/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

input_dir = pyd.Directory(in_dir).InputDIR()

class Voxelize:
    def __init__(self, directory, booltype = False):
        self.directory = directory
        self.booltype = booltype

    def toNumpy(self):
        res = np.asarray(Image.open(self.directory+os.listdir(self.directory)[0])).shape
        arr = np.empty([res[0], res[1], len(os.listdir(self.directory))])
        res1 = arr.shape
        # print(arr.shape)
        for i,j in enumerate(os.listdir(self.directory)):
            arr[:,:,i] = np.asarray(Image.open(self.directory+j), dtype=bool)
        return arr, res1
    
    def coordArray(self):
        arr = self.toNumpy()[0]
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]), 3), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i] = coord[:,i]
        if self.booltype == False:
            return coords
        else:
            return self.npArray()

    def matpropsArray(self):
        # Stores properties of points using an array of same length as coord array. 
        # 4 can be changed to a different number depending on what needs to be stored.
        carray = self.coordArray()
        mparr = np.empty((np.shape(carray)[0], 4), dtype=int)
        return mparr

    def npArray(self):
        cArray = self.coordArray()[0]
        res = self.toNumpy()[1]
        nArray = np.empty(shape=(res), dtype=bool)
        for i in cArray:
            nArray[i[0]][i[1]][i[2]] = True
        return nArray

# write class which calculates degradation based on data from voxelise

def delPixel():
    pass
    
def neighbours(x, y, z, res):
    #take coordinates, calculate surrounding coordinates -> return a list of coordinates
    #create a list of x coordinates
    x_range = np.arange(x-1, x+2)
    y_range = np.arange(y-1, y+2)
    z_range = np.arange(z-1, z+2)
    n_list = []
    for i in x_range:
        for j in y_range: #check if coordinates are negative or larger than image
            for k in z_range:
                if -1 < x <= res[0] and -1 < y <= res[1] and -1 < z <= res[2]:
                    if (x != i or y != j or z != k) and (0 <= i <= res[0]) and (0 <= j <= res[1]) and (0 <= k <= res[2]):
                        n_list.append([i, j, k])
    return n_list

a = Voxelize(input_dir)
# a.coordArray()
# print(a.coordArray()[0])
# arr_shape = Voxelize(input_dir).toNumpy()[1]
# print(arr_shape)
# n = neighbours(2,2,2, arr_shape)
# print(n, len(n))

class PixelClassifier():
    def __init__(self, coords_array, prop_array):
        self.coords_array = coords_array
        self.prop_array = prop_array

    def initClassify(self, chi):
        #prop_array[n][0] = x_i,j
        #x_i,j = 0 -> amorphous, x_i,j = 1 -> crystalline,
        #x_i,j = -1 -> eroded
        #prop_array[n][1] = s
        #prop_array[n][2] = molecular weight
        #prop_array[n][3] = pseudo-elastic modulus
        for i in self.prop_array:
            i[0] = self.crystallinity(chi)
            i[1] = self.initpixelState(chi)
        return self.prop_array
    
    def crystallinity(self, chi):
        #crystallinity = probability a pixel will be crystalline
        bin_prob = np.random.binomial(1, chi)
        if bin_prob == 1:
            x_chi = 1
        else: 
            x_chi = 0
        return x_chi
    
    def initpixelState(self, chi):
        crys = self.crystallinity(chi)
        if crys == -1:
            s = 0
        elif crys == 1:
            s = 1
        elif crys == 0:
            s = 1
        else:
            raise ValueError('Undefined pixel state.')
        return s

    def molecularWeight(self, mw0):
        pass
    
    def update_pixel(self):
        pass
    
def binomial(n, p, size=None):
    return np.random.binomial(n, p, size)
print(binomial(1, 0.3))

def comment():
    '''
    # Pass xyz to Open3D.otd.geometry.PointCloud and visualize

    pcd = otd.geometry.PointCloud()
    pcd.points = otd.utility.Vector3dVector(a)
    # otd.io.write_point_cloud("sync.ply", pcd)
    otd.visualization.draw_geometries([pcd])

    # vox = otd.geometry.VoxelGrid()
    # vox.create_from_point_cloud(1, pcd.points)

    # Load saved point cloud and visualize it
    # pcd_load = otd.io.read_point_cloud("sync.ply")
    # otd.visualization.draw_geometries([pcd_load])

    '''
    '''
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.otd.geometry.PointCloud and visualize
    pcd = otd.geometry.PointCloud()
    pcd.points = otd.utility.Vector3dVector(xyz)
    otd.io.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = otd.io.read_point_cloud("sync.ply")
    otd.visualization.draw_geometries([pcd_load])

    '''