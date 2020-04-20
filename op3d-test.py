import pydirectory as pyd
import open3d as otd
import numpy as np
# import stl
import numpy
import random
import scipy as sp
import scipy.ndimage as ndimage
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import pyntcloud


in_dir = "/data/sample1/25/uct/tiff/" #'/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

input_dir = pyd.Directory(in_dir).InputDIR()

class Voxelize:
    def __init__(self, directory, booltype = False):
        self.directory = directory
        self.booltype = booltype

    def toNumpy(self):
        """[summary]
        
        Returns:
            array, array -- converts image to numpy array and returns the array and its size.
        """
        res = np.asarray(Image.open(self.directory+os.listdir(self.directory)[0])).shape
        arr = np.empty([res[0], res[1], len(os.listdir(self.directory))])
        res1 = arr.shape
        # print(arr.shape)
        for i,j in enumerate(os.listdir(self.directory)):
            arr[:,:,i] = np.asarray(Image.open(self.directory+j), dtype=bool)
        return arr, res1
    
    def coordArray(self):
        """[summary]
        
        Returns:
            array -- that contains the coordinates of the pixels that are white.
        """
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
        """[summary]
        [chi, pixel state, e, prop]

        Can make an input to decide how many properties are stored,
        e.g. matpropsArray(self, numprops)

        Returns:
            array -- empty array of the length of coordArray that containsvarious material properties.
        """
        # Stores properties of points using an array of same length as coord array. 
        # 4 can be changed to a different number depending on what needs to be stored.
        carray = self.coordArray()
        return np.empty((np.shape(carray)[0], 4), dtype=int)

    def npArray(self):
        """[summary]
        
        Returns:
            array -- identical to toNumyp() but array is boolean.
        """
        cArray = self.coordArray()
        # print(cArray)
        res = self.toNumpy()[1]
        # print(res)
        nArray = np.empty(shape=(res), dtype=bool)
        # print(nArray)
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
                if (
                    -1 < x <= res[0]
                    and -1 < y <= res[1]
                    and -1 < z <= res[2]
                    and (x != i or y != j or z != k)
                    and (0 <= i <= res[0])
                    and (0 <= j <= res[1])
                    and (0 <= k <= res[2])
                ):
                    n_list.append([i, j, k])
    return n_list

def randScalar(min, max):
    return min + (random.random() * (max - min))

a = Voxelize(input_dir)
coords = a.coordArray()
props = a.matpropsArray()
nparr = a.npArray()
# print(a.coordArray())
# print(a.npArray())
# arr_shape = Voxelize(input_dir).toNumpy()[1]
# print(arr_shape)
# n = neighbours(2,2,2, arr_shape)
# print(n, len(n))
class InitPixelClassifier():
    """Object Class that initialises the properties of the voxel model.
    """
    def __init__(self, coords_array, np_array, prop_array):
        self.coords_array = coords_array
        self.np_array = np_array
        self.prop_array = prop_array

    def initClassify(self, chi, mw0, e):
        #prop_array[n][0] = x_i,j
        #x_i,j = 0 -> amorphous, x_i,j = 1 -> crystalline,
        #x_i,j = -1 -> eroded
        #prop_array[n][1] = s
        #prop_array[n][2] = molecular weight
        #prop_array[n][3] = pseudo-elastic modulus
        for i in self.prop_array:
            i[0] = self.crystallinity(chi)
            i[1] = self.initpixelState(chi)
            i[2] = self.initModulus(e)
            i[3] = self.initMolecularWeight(mw0)
        return self.prop_array
    
    def crystallinity(self, chi):
        #crystallinity = probability a pixel will be crystalline
        if type(chi) == float:
            bin_prob = np.random.binomial(1, chi)
            return 1 if bin_prob == 1 else 0
        elif type(chi) == int:
            self.crystallinity(float(chi))

            print('Ensure chi is float.')

    
    def initpixelState(self, chi):
        crys = self.crystallinity(chi)
        if crys == -1:
            s = 0
        elif crys in [1, 0]:
            s = 1
        else:
            raise ValueError('Undefined pixel state.')
        return s

    def initMolecularWeight(self, mw0):
        return mw0
    
    def initModulus(self, e):
        e_scalar = randScalar(0.9, 1)
        return e * e_scalar

scaffold = InitPixelClassifier(coords, nparr, props)
scaffold.initClassify(0.8, 10000, 3.4)
# print(coords)
# print(nparr)
# print(props)
    
def binomial(n, p, size=None):
    return np.random.binomial(n, p, size)
# print(binomial(1, 0.3))

class UpdateModel():
    """Object Class that updates the properties of the voxel model.
    """
    def __init__(self, coords_array, np_array, prop_array):
        self.coords_array = coords_array
        self.np_array = np_array
        self.prop_array = prop_array
    
    def update(self):
        self.updtCrystallinity()
        self.updtPixelState()
        self.updtMolecularWeight()
        self.updtModulus()
        pass

    def updtCrystallinity(self, chi):
        edt = ndimage.distance_transform_edt(self.np_array)
        #check edt layer by layer, if edt <= 2 (pixel beside void) --> update its crystallinity

        pass

    def updtPixelState(self):
        crys = None
        pass

    def updtMolecularWeight(self):
        pass

    def updtModulus(self):
        pass



# md = pyntcloud
# Pass xyz to Open3D.otd.geometry.PointCloud and visualize

pcd = otd.geometry.PointCloud()
pcd.points = otd.utility.Vector3dVector(a.coordArray())
# otd.io.write_point_cloud("sync.ply", pcd)

#from o3d to pyntcloud
cloud = pyntcloud.from_instance("open3d", pcd.points)
print(cloud)

otd.visualization.draw_geometries([pcd])

# vox = otd.geometry.VoxelGrid()
# vox.create_from_point_cloud(1, pcd.points)

# Load saved point cloud and visualize it
# pcd_load = otd.io.read_point_cloud("sync.ply")
# otd.visualization.draw_geometries([pcd_load])

# x = np.linspace(-3, 3, 401)
# mesh_x, mesh_y = np.meshgrid(x, x)
# z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
# z_norm = (z - z.min()) / (z.max() - z.min())
# xyz = np.zeros((np.size(mesh_x), 3))
# xyz[:, 0] =  np.reshape(mesh_x, -1)
# xyz[:, 1] = np.reshape(mesh_y, -1)
# xyz[:, 2] = np.reshape(z_norm, -1)
# print('xyz')
# print(xyz)

# Pass xyz to Open3D.otd.geometry.PointCloud and visualize
# pcd = otd.geometry.PointCloud()
# pcd.points = otd.utility.Vector3dVector(xyz)
# otd.io.write_point_cloud("sync.ply", pcd)

# Load saved point cloud and visualize it
# pcd_load = otd.io.read_point_cloud("sync.ply")
# otd.visualization.draw_geometries([pcd_load])

