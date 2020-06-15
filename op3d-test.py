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
from pyntcloud import PyntCloud
import pyvista as pv
import PVGeo as pg

in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

out_dir = "/data/deg_test_output/"

dir = pyd.Directory(in_dir, out_dir)

input_dir = dir.InputDIR()

output_dir = dir.OutputDIR()

new_input_dir = pyd.Directory(out_dir).InputDIR()

print(input_dir)
print(output_dir)


class Voxelize:
    def __init__(self, directory, booltype=False):
        self.directory = directory
        self.booltype = booltype

    def to_numpy(self):
        """[summary]
        Returns:
            array, array -- converts image to numpy array and returns the array and its size.
        """
        res = np.asarray(Image.open(self.directory + os.listdir(self.directory)[0])).shape
        arr = np.empty([res[0], res[1], len(os.listdir(self.directory))])
        res1 = arr.shape
        # print(arr.shape)
        for i, j in enumerate(os.listdir(self.directory)):
            arr[:, :, i] = np.asarray(Image.open(self.directory + j), dtype=bool)
        return arr, res1

    def coord_array(self):
        """[summary]
        
        Returns:
            array -- that contains the coordinates of the pixels that are white.
        """
        arr = self.to_numpy()[0]
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]), 3), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i] = coord[:, i]
        if not self.booltype:
            return coords
        else:
            return self.np_array()

    def matprops_array(self, numprops):
        """[summary]
        [chi, pixel state, lifetime, adjacent, e, molecular weight]

        Can make an input to decide how many properties are stored,
        e.g. matprops_array(self, numprops)

        Returns:
            array -- empty array of the length of coord_array that containsvarious material properties.
        """
        # Stores properties of points using an array of same length as coord array. 
        # 4 can be changed to a different number depending on what needs to be stored.
        carray = self.coord_array()
        return np.empty((np.shape(carray)[0], numprops), dtype=int)

    def np_array(self):
        """[summary]
        
        Returns:
            array -- identical to toNumyp() but array is boolean.
        """
        cArray = self.coord_array()
        # print(cArray)
        res = self.to_numpy()[1]
        # print(res)
        nArray = np.empty(shape=(res), dtype=bool)
        # print(nArray)
        for i in cArray:
            nArray[i[0]][i[1]][i[2]] = True
        print(np.shape(nArray))
        return nArray


# write class which calculates degradation based on data from voxelise

def delPixel():
    pass


def neighbours(x, y, z, res):
    # take coordinates, calculate surrounding coordinates -> return a list of coordinates
    # create a list of x coordinates
    x_range = np.arange(x - 1, x + 2)
    y_range = np.arange(y - 1, y + 2)
    z_range = np.arange(z - 1, z + 2)
    n_list = []
    for i in x_range:
        for j in y_range:  # check if coordinates are negative or larger than image
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


def rand_scalar(min_val, max_val):
    return min_val + (random.random() * (max_val - min_val))


a = Voxelize(input_dir)
coords = a.coord_array()
props = a.matprops_array(numprops=6)
nparr = a.np_array()


def adjacent(index, coords_array, loc_arr):
    """
    1 = adjacent
    0 = far
    """
    a = np.where((loc_arr == coords_array[index]).all(axis=1))[0]
    # print('a is ', a)
    if a.size:
        # print('not empty')
        return 1, a
    else:
        return 0, a


# print(a.coord_array())
# print(a.np_array())
# arr_shape = Voxelize(input_dir).to_numpy()[1]
# print(arr_shape)
# n = neighbours(2,2,2, arr_shape)
# print(n, len(n))
class InitPixelClassifier:
    """Object Class that initialises the properties of the voxel model.
    """

    def __init__(self, coords_array, np_array, prop_array):
        self.coords_array = coords_array
        self.np_array = np_array
        self.prop_array = prop_array

    def init_classify(self, chi, mw0, e):
        '''
        Add False values to 3D numpy array surrounding it.
        shp_x, shp_y, shp_z = np.shape(self.np_array)
        surr_arr = np.zeros((shp_x, shp_y), dtype=np.bool)
        print(surr_arr.shape)
        print(np.full((shp_x, 1), False).shape)
        print(self.np_array[:,:,0].shape)
        print(self.np_array.shape)
        for i in range(self.np_array.shape[2]):
            np.insert(self.np_array[:,:,i], 0, False, axis=0)
            print(self.np_array[:,:,i])
            # i = np.insert(i, 0, False, axis=0)
            # i = np.append(i, np.full((shp_x, 1), False), axis=1)
            # print(i)
        '''

        edt = ndimage.distance_transform_edt(self.np_array)
        loc = np.where(edt == 1)
        loc_arr = np.vstack((loc[0], loc[1], loc[2])).transpose()
        # self.init_adjacent(loc_arr)

        # prop_array[n][0] = x_i,j
        # x_i,j = 0 -> amorphous, x_i,j = 1 -> crystalline,
        # x_i,j = -1 -> eroded
        # prop_array[n][1] = s
        # prop_array[n][2] = molecular weight
        # prop_array[n][3] = pseudo-elastic modulus
        for i, j in enumerate(self.prop_array):
            j[0] = self.init_crystallinity(chi)
            j[1] = self.init_pixel_state(chi)
            j[2] = self.init_life()
            # print(i)
            j[3] = adjacent(i, self.coords_array, loc_arr)[0]
            # j[3] = self.init_adjacent(i, loc_arr)
            j[4] = self.init_modulus(e)
            j[5] = self.init_molecular_weight(mw0)
        return self.prop_array, loc_arr

    def init_crystallinity(self, chi):
        # crystallinity = probability a pixel will be crystalline
        bin_prob = np.random.binomial(1, chi)
        return 1 if bin_prob == 1 else 0


    def init_pixel_state(self, chi):
        '''
        Initially all pixels are on (i.e. state = 1)
        '''
        # crys = self.init_crystallinity(chi)
        # if crys == -1:
        #     s = 0
        # elif crys in [1, 0]:
        #     s = 1
        # else:
        #     raise ValueError('Undefined pixel state.')
        return 1

    def init_life(self):
        '''
        -1 = infinite life (only initially) --> once in contact with environment (at t = 0)
        update with random probability (life is in days, i.e. at t=0, voxels with life = 10 -> will last for 10 days)
        only surface elements have a finite life, internal voxels still have life = -1
        '''
        return -1

    def init_adjacent(self, index, loc_arr):
        '''
        1 for adjacent
        0 for not adjacent
        voxels on the edge are not marked adjacent. needs to be fixed.
        '''
        a = np.where((loc_arr == self.coords_array[index]).all(axis=1))[0]
        if a.size:
            # print('not empty')
            return 1
        else:
            return 0
        # for i in loc_arr:
        #     print(i)
        #     if np.array_equal(self.coords_array[index], i):
        #         print(i)
        #         print(self.prop_array[index])
        #         return 1
        #     else:
        #         return 0
        # for i, j in enumerate(self.coords_array):
        #     for k in loc_arr:
        #         if np.array_equal(j, k) is True:
        #             return 1

    def init_molecular_weight(self, mw0):
        return mw0

    def init_modulus(self, e):
        e_scalar = rand_scalar(0.9, 1)
        return e * e_scalar


scaffold = InitPixelClassifier(coords, nparr, props)
scaffold.init_classify(0.65, 10000, 3.4)


def binomial(n, p, size=None):
    return np.random.binomial(n, p, size)


class UpdateModel:
    """
    Object Class that updates the properties of the voxel model.
    """

    def __init__(self, n, coords_array, np_array, prop_array):
        self.n = n
        self.coords_array = coords_array
        self.np_array = np_array
        self.prop_array = prop_array

    def update(self):
        self.updt_life(self.n)
        self.updt_molecular_weight()
        self.updt_modulus()
        self.updt_loc_array()
        return self.np_array, self.prop_array, self.coords_array

    def updt_loc_array(self):
        edt = ndimage.distance_transform_edt(self.np_array)
        loc = np.where(edt == 1)
        loc_arr = np.vstack((loc[0], loc[1], loc[2])).transpose()
        # print(loc_arr)
        for i, j in enumerate(self.prop_array):
            j[3] = adjacent(i, self.coords_array, loc_arr)[0]
        return self.prop_array

    def updt_crystallinity(self):
        '''
        Crystallinity doesn't change of voxels (this function is invalid), kept here for reference
        '''
        edt = ndimage.distance_transform_edt(self.np_array)
        counter = 0
        while counter < 1:  # np.shape(self.np_array)[2]:
            # if any(edt[:, :, counter]) <= 2:
            loc = np.where(edt[:, :, counter] <= 2)  # finds location of adjacent voxels
            loc_x = loc[0]
            loc_y = loc[1]
            loc_arr = np.insert(np.vstack((loc_x, loc_y)).transpose(), 2, counter, axis=1)
            for i, j in enumerate(self.coords_array):
                for k in loc_arr:
                    if np.array_equal(j, k) is True:
                        if self.prop_array[i][0] == 0:  # check for amorphous
                            self.prop_array[i][1] = 0
                        elif self.prop_array[i][0] == 1:
                            self.prop_array[i][1] = 1

            break

    def updt_pixel_state(self, index):
        """
        index is for location that is deleted
        """
        vox_x, vox_y, vox_z = self.coords_array[index]
        # print(vox_x, vox_y, vox_z)
        # old_arr = self.np_array
        self.np_array[vox_x][vox_y][vox_z] = False

        # print(self.np_array[vox_x][vox_y][vox_z])
        # print(np.array_equal(old_arr, self.np_array))
        # self.coords_array = np.delete(self.coords_array, index, axis=0)
        # self.prop_array = np.delete(self.prop_array, index, axis=0)

    def updt_life(self, n):
        # print('ieration ', n)
        lam_a = 25
        lam_c = 50
        for i, j in enumerate(self.prop_array):
            # check if adjacent
            if j[3] == 1:
                # check if infinite life
                if j[2] < 0:
                    # check if crystalline
                    if j[1] == 1:
                        # add life
                        j[2] = np.random.poisson(lam_c)
                        # print('life added c ', j[2])
                    else:
                        j[2] = np.random.poisson(lam_a)
                        # print('life added a ', j[2])
                elif n > 0 and j[2] > 0:
                    j[2] -= 1
                    # print('life remaining ', j[2])
                elif j[2] == 0:
                    j[1] = 0
                    j[3] = 0
                    self.updt_pixel_state(i)

    def updt_molecular_weight(self):
        pass

    def updt_modulus(self):
        pass


class Poisson:
    def __init__(self, lam, num_days):
        self.lam = lam
        self.num_days = num_days

    def poisson(self):
        return np.random.poisson(self.lam, size=self.num_days)


s = np.random.poisson(15, 10)
# print(s)

# b = UpdateModel(coords, nparr, props)
# cds = pd.DataFrame(coords, columns=['x', 'y', 'z'])
# print(cds)
# md = PyntCloud(cds)
# md.plot()
# mdotd = md.to_instance(library='open3d')
# print(mdotd)

week = 14*8
newcoord = np.load(str(new_input_dir)+'coord_arr_'+str(week)+'.npy')
newmatprops = np.load(str(new_input_dir)+'prop_arr_'+str(week)+'.npy')
newnp = np.load(str(new_input_dir)+'np_arr_14.npy')
print(newcoord.shape, newmatprops.shape)
del_list = []
for i, j in enumerate(newmatprops):
    if j[1] == 0:
        del_list.append(i)
newmatprops = np.delete(newmatprops, del_list, axis=0)
newcoord = np.delete(newcoord, del_list, axis=0)
print(newcoord.shape, newmatprops.shape)
ptcloud = pg.points_to_poly_data(newcoord)
# ptcloud = pv.PolyData(newcoord)
print(ptcloud)
# print(ptcloud)
data = newmatprops[:, 0]
ptcloud['Crystallinity'] = data
# ptcloud.plot()
voxelizer = pg.filters.VoxelizePoints()
grid = voxelizer.apply(ptcloud)
grid.plot()

# save npy file every N iteration
#
# n = np.arange(0, 154, 14)
#
# for i in range(142):
#     updtmod = UpdateModel(i, coords, nparr, props).update()
#     print(i)
#     if i in n:
#         np.save(str(output_dir + 'np_arr' + '_' + str(i)), updtmod[0])
#         np.save(str(output_dir + 'prop_arr' + '_' + str(i)), updtmod[1])
#         np.save(str(output_dir + 'coord_arr' + '_' + str(i)), updtmod[2])



    # if np.any(updtmod[0] == True):
    #     print('True voxels present')
# grid.plot()
# Pass xyz to Open3D.otd.geometry.PointCloud and visualize

# pcd = otd.geometry.PointCloud()
# pcd.points = otd.utility.Vector3dVector(mdotd) #(a.coord_array())
# otd.io.write_point_cloud("sync.ply", pcd)

# from o3d to pyntcloud
# cloud = pyntcloud.from_instance("open3d", pcd.points)
# print(cloud)

# otd.visualization.draw_geometries([pcd])

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
