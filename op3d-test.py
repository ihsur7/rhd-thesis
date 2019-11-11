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
    
    def coordArray(self):
        arr = self.toNumpy()
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]), 3), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i] = coord[:,i]
        return coords

    #add method that takes coordinates from coordArray and turns it back to numpy array
    def npArray(self):
        pass
    
a = Voxelize(input_dir).coordArray()

# Pass xyz to Open3D.otd.geometry.PointCloud and visualize

pcd = otd.geometry.PointCloud()
pcd.points = otd.utility.Vector3dVector(a)
otd.io.write_point_cloud("sync.ply", pcd)

vox = otd.geometry.VoxelGrid()
vox.create_from_point_cloud(1, pcd.points)

# Load saved point cloud and visualize it
pcd_load = otd.io.read_point_cloud("sync.ply")
otd.visualization.draw_geometries([pcd_load])

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
# points = pd.DataFrame(a)
# print(points)

# z,x,y = a.nonzero()
# b = a.toNumpy()
# print(b.shape)
# c = a.NumpytoVtk(b)
# print(c)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.voxels(a, edgecolor='k')
# plt.show()