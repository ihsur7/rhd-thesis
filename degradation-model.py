from sys import path
import pydirectory as pyd
# import open3d as otd
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
import math
import uuid
from tqdm import tqdm

print('numpy version = ', np.__version__)

in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

out_dir = "/data/deg_test_output/"

dir = pyd.Directory(in_dir, out_dir)

input_dir = dir.InputDIR()

output_dir = dir.OutputDIR()

new_input_dir = pyd.Directory(out_dir).InputDIR()

class Presets:
    def __init__(self, led):
        self.led = led
        self.preset = {0.135: {"mw_loss_coeff": 1369}, \
                    0.097: {"mw_loss_coeff": 1220}, \
                    0.077: {"mw_loss_coeff": 1034}}

    def mw_loss_coeff(self):
        return self.preset[self.led]["mw_loss_coeff"]
    

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
            array -- that contains coordinates of the pixels that are white.
        """
        arr = self.to_numpy()[0]
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]), 4), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i][0] = i
            coords[i][1:4] = coord[:, i]

        # print(coords)
        if not self.booltype:
        # print(self.prop_array.shape)
            # print(coords.shape)
            # l = self.prop_array.shape[0]
            # self.coords_array = np.c_[np.zeros(l), self.coords_array]
            return coords
        else:
            return self.np_array()
    
    def vox_id(self):
        arr1 = self.to_numpy()
        arr = arr1[0]
        res = arr1[1]
        coords = self.coord_array()
        for i in coords:
            x,y,z = i[1:4]
            arr[x,y,z] = i[0]
            # print(i[1:4])
            # arr[i[1:4]] == i[0]
        return arr, res

    def matprops_array(self, numprops):
        """[summary]
        [id, nth pixel, pixel state, adjacent, prob, water diffusion, chi, molecular weight, e]
        #[chi, pixel state, lifetime, adjacent, e, molecular weight]
        diffusion = initially false
        Can make an input to decide how many properties are stored,
        e.g. matprops_array(self, numprops)

        Returns:
            array -- empty array of the length of coord_array that contains various material properties.
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
        # print(np.shape(nArray))
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
                        -1 < x <= res[0]-1
                        and -1 < y <= res[1]-1
                        and -1 < z <= res[2]-1
                        and (x != i or y != j or z != k)
                        and (0 <= i <= res[0]-1)
                        and (0 <= j <= res[1]-1)
                        and (0 <= k <= res[2]-1)
                ):
                    n_list.append([i, j, k])
    return n_list


def rand_scalar(min_val, max_val):
    return min_val + (random.random() * (max_val - min_val))


def adjacent(index, coords_array, loc_arr):
    """
    1 = adjacent
    0 = far
    """
    # print(coords_array[index][0:3])
    a = np.where((loc_arr == coords_array[index][1:4]).all(axis=1))[0]
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

    def init_classify(self, led, chi, e):
        '''
        [id, is_pathed?, pixel state, adjacent, water concentration, is crystalline, molecular weight, e]

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

        edt = ndimage.distance_transform_edt(self.np_array[0])
        loc = np.where(edt == 1)
        loc_arr = np.vstack((loc[0], loc[1], loc[2])).transpose()
        # print(self.coords_array)
        # self.init_adjacent(loc_arr)

        # prop_array[n][0] = x_i,j
        # x_i,j = 0 -> amorphous, x_i,j = 1 -> crystalline,
        # x_i,j = -1 -> eroded
        # prop_array[n][1] = s
        # prop_array[n][2] = molecular weight
        # prop_array[n][3] = pseudo-elastic modulus
        for i, j in enumerate(self.prop_array):
            j[0] = self.init_id(i)
            j[1] = self.init_ispathed()
            j[2] = self.init_pixel_state()
            j[3] = adjacent(i, self.coords_array, loc_arr)[0]
            j[4] = self.init_diffusion()
            j[5] = self.init_crystallinity(chi)
            j[6] = self.init_molecular_weight(led)
            j[7] = self.init_modulus(e)
            # j[4] = self.init_prob(i, self.coords_array, self.np_array, bias = True)

            # print(i)
            # j[3] = self.init_adjacent(i, loc_arr)
        return self.prop_array, loc_arr
    
    def init_id(self, index):#, index):
        # return [i for i in self.coords_array[index]]
        # for i, j in enumerate(self.prop_array):
        #     j[0] = i
        # self.coords_array[index][0] = index
        return index

    # def init_prob(self, index, coord_array, np_array, bias = True):
    #     # adj_array = adjacent(index, self.coords_array, loc_arr)
    #     if bias == False:
    #         return 1/27
    #     else:
    #         if self.prop_array[index][6] == 0: #if amorphous
    #             x, y, z = coord_array[index][0], coord_array[index][1], coord_array[index][2]
    #             neighbour_list = neighbours(x, y, z, res = np.array[1])
    #             for i in neighbour_list:

    #             return 1/27
    #         else
    
    def init_ispathed(self):
        return 0

    def init_diffusion(self):
        return 0.0

    def init_crystallinity(self, chi):
        # crystallinity = probability a pixel will be crystalline
        bin_prob = np.random.binomial(1, chi)
        return 1 if bin_prob == 1 else 0

    def init_pixel_state(self):
        '''
        Initially all pixels are on (i.e. state = 1)
        0 = off/dead
        1 = on
        2 = active (only active if neighbouring pixel active, adjacent pixels are active)
        '''
        # crys = self.init_crystallinity(chi)
        # if crys == -1:
        #     s = 0
        # elif crys in [1, 0]:
        #     s = 1
        # else:
        #     raise ValueError('Undefined pixel state.')
        return 1

    # def init_life(self):
    #     '''
    #     -1 = infinite life (only initially) --> once in contact with environment (at t = 0)
    #     update with random probability (life is in days, i.e. at t=0, voxels with life = 10 -> will last for 10 days)
    #     only surface elements have a finite life, internal voxels still have life = -1
    #     '''
    #     return -1

    def init_adjacent(self, index, loc_arr):
        '''
        1 for adjacent
        0 for not adjacent
        voxels on the edge are not marked adjacent. needs to be fixed.
        '''
        a = np.where((loc_arr == self.coords_array[index][1:3]).all(axis=1))[0]
        if a.size:
            # print('not empty')
            return 1
        else:
            return 0
        


    def init_molecular_weight(self, led):
        if led == 0.135:
            mw0 = np.random.poisson(390000)
        elif led == 0.097:
            mw0 = np.random.poisson(460000)
        elif led == 0.077:
            mw0 = np.random.poisson(510000)
        else:
            print("Unknown LED value.")
            return
        return mw0

    def init_modulus(self, e):
        e_scalar = rand_scalar(0.9, 1)
        return e * e_scalar
    
def update_model():
    i = 0
    # 1 iteration = 1 second
    max_iter = 100
    while i <= max_iter:
        i += 1
        if bool(coords[3]) == True:
            # Fick(diff_coeff_mm["37"], )
            break
    return

class PathArray:
    def __init__(self, coords_array, np_array, prop_array):
        self.coords_array = coords_array
        self.np_array = np_array
        self.prop_array = prop_array

    def init_prob(self, index, coords_array, np_array, bias = True):
        probarray = np.zeros(shape = (self.prop_array.shape[0], 2))
        print(probarray)
        # adj_array = adjacent(index, self.coords_array, loc_arr)
        if bias == False:
            return 1/27
        else:
            if self.prop_array[index][6] == 0: #if amorphous
                x, y, z = coords_array[index][0], coords_array[index][1], coords_array[index][2]
                neighbour_list = neighbours(x, y, z, res = np_array[1])
                for i in neighbour_list:

                    pass
                return 1/27
            else:
                pass

    def initPathArray(self):
        """
        Returns:
            array of (n,1) dimensions containing ID of voxels starting with the first exposed voxel
        """
        # print(self.prop_array)
        patharray = []#np.zeros(shape=(np.count_nonzero(self.prop_array[:,3]), 1), dtype=int)
        # print(np.count_nonzero(self.prop_array[:,3]))
        # print(patharray, self.prop_array.shape)
        # for x,y in enumerate(patharray):
        #     for i in self.prop_array:
        #         if i[3] == 1:
        #             y = i[0]
        #             patharray[x] = i[0]
        # print(self.prop_array[:,3])
        for i in self.prop_array:
            if i[3] == 1:
                # print(i)
                patharray.append(i[0])
                i[1] = 1
                # counter += 1   
            # for j in self.prop_array:
            #     if j[3] == 1:
            #         i[0] = j[0]
        # print(len(patharray))
        patharray = np.asarray_chkfinite(patharray)
        # print(patharray.shape)
        patharray = patharray.reshape(patharray.shape[0],-1)
        # print(patharray.shape)
        # print(patharray)
        return patharray
    
    def initRandomWalk(self):
        patharr = self.initPathArray()

        pass
    

def random_walk(pixel_state):
    dims = 3
    if pixel_state != 2:
        return None
    step_set = [-1, 0, 1]
    return np.random.choice(a=step_set, size=(1, dims))

a = Voxelize(input_dir)
coords = a.coord_array()
props = a.matprops_array(numprops=8)
nparr = a.to_numpy()
idarr = a.vox_id()
mat_props = InitPixelClassifier(coords, nparr, props).init_classify(0.135, 0.67, 3)
# print(mat_props[0])
flowpath = PathArray(coords, nparr, mat_props[0]).initPathArray()
# print(flowpath)
# prob_array = np.zeros(shape=nparr[1], dtype=float)
# print(nparr[1])
# print(prob_array, prob_array.shape)
# prob_array = np.zeros(shape=(3,3,3), dtype=float)
# print(prob_array)
# print('end', coords)
def iter(coords_array, prop_array, np_array, id_array, path_array, bias = False):
    #initialise 3D array shape of np_array with zeros that dynamically changes values
    #the values are probability numbers 
    # prob_array = np.zeros(shape=np_array[1], dtype=float)
    # prob_array = np.zeros(shape=(3,3,3), dtype=float)
    # print(prob_array)

    #for each property array row, if voxel is active, run randomwalk
    t = 0
    print("t = ", t)

    max_steps = 3
    prob_crys = 0.5
    prob_amorph = 1.5
    prob_self = 0.5
    # print(coords_array)
    # print(prop_array[0])
    coords_dict = {}
    prop_dict = {}
    prop_shape = prop_array[0].shape
    # print(path_array.shape)
    # print(prop_shape)
    for i in coords_array:
        coords_dict[i[0]] = np.array(i[1:4])
    for j in prop_array[0]:
        prop_dict[j[0]] = j[1:prop_shape[1]]
    # print(prop_dict)
    # print(mat_props)
    # print(path_array)
    # print(coords_dict[0])
    # for t in tqdm(range(max_steps)):
    # pbar = tqdm(total=max_steps+1)
    while t < max_steps:
        path_array = np.c_[path_array, np.zeros(path_array.shape[0])]
        it = 0
        #iterates through coordinate array, instead it should iterate through flowpath array as it needs assign the next coordinate for the path
        for j in coords_array:
            # print(t)
            # print(path_array.shape)
            # print(j)
            #checks if voxel is active, find neighbouring voxels
            if prop_dict[j[0]][2] == 1:
                # print(j)
                it += 1
            # print(k)
            # if k[2] == 1: 
                # print(coords_array)
                x,y,z = coords_dict[j[0]]
                # x, y, z = coords_array[j][1], coords_array[j][2], coords_array[j][3]
                #gets coordinates of neighbouring voxels
                neighbour_list = neighbours(x, y, z, res = np_array[1])
                # print(neighbour_list)
                id_list = []
                id_list.append(j[0])
                for a in neighbour_list:
                    #if pixel is white/polymer
                    # print(a)
                    x1, y1, z1 = a[0], a[1], a[2]
                    # print(np_array[0].shape)
                    # print(np_array[0][x1][y1][z1])
                    # print(x1, y1, z1)
                    if z1 == 56:
                        print(neighbour_list)
                        print(a)
                        print(z1)
                    # print(np_array[0][:,:,56])
                    
                    # print('shape = ', np_array[0].shape)
                    if np_array[0][x1][y1][z1] == 1: #if pixel is white
                        #get coordinate id
                        
                        id_list.append(id_array[0][x1][y1][z1]) #np.array([x1, y1, z1])
                        # print(id_list)
                        id_list = [int(i) for i in id_list]
                        # loc = np.where((coords_array[:,1:4] == coord).all(axis=1))
                        # print(loc[0])
                        # for ii in loc[0]:
                        #     id_list.append(coords_array[ii,0])
                        # print("idlist = ",id_list)
                        # loc = np.where((coords_array[1:3] == np.array(a)).all(axis=1))
                        #count number of pixels
                # print('it = ', it)
                # print('idlist = ', id_list)
                countamorph = [] #list containing id of voxels that are neighbouring to the current voxel and are amorphous
                countcrys = [] #same as above, except for crystalline
                #find if voxel is amorphous of crystalline
                # print(mat_props[0][1][5])
                for iid in id_list:
                    if prop_dict[iid][5] == 1:
                        countcrys.append(iid)
                    # if mat_props[0][iid][5] == 1: #if crystalline
                        # countcrys.append(iid)
                    else:
                        countamorph.append(iid)
                total_count = len(countcrys) + len(countamorph)
                # print(total_count, len(countamorph), len(countcrys))
                #create a 3D array with current voxel in the middle that is 3x3 and add neighbouring voxels to this array
                prob_matrix = np.zeros(shape=(3,3,3), dtype=float)
                key_matrix = np.zeros(shape=(3,3,3), dtype=int)
                key_matrix[1,1,1] = id_list[0]
                # print(prob_matrix)
                # print(key_matrix)
                prob_matrix[1,1,1] = prob_self*(1/total_count)
                center_id = id_list[0]
                center_coord = coords_dict[center_id]
                center_coord_x = center_coord[0]
                center_coord_y = center_coord[1]
                center_coord_z = center_coord[2]
                # print(id_list[1:])
                for i in id_list[1:]:
                    # print(i)
                    xx = coords_dict[i]#coords_array[np.where(coords_array[:,0] == i)]
                    # print(xx)
                    x_x = center_coord_x - xx[0]
                    x_y = center_coord_y - xx[1]
                    x_z = center_coord_z - xx[2]
                    # x_list = [x_x, x_y, x_z]
                    key_matrix[1-x_x, 1-x_y, 1-x_z] = i
                    # if i == center_id:
                    #     prob_matrix[1, 1, 1] = prob_self*(1/total_count)
                    if bias == False:
                        prob_matrix[1-x_x, 1-x_y, 1-x_z] = (1-(prob_self/total_count))/(total_count-1)
                            # print(prob_matrix)
                            # print(key_matrix)
                    else:
                        if i in countamorph:
                            # if i == center_id:
                            #     pass
                            # else:
                            if prop_dict[i][5] == 0:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph)-1)
                            else:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph))
                        elif i in countcrys:
                            # if i == center_id:
                            #     pass
                            # else:
                            if prop_dict[i][5] == 1:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)-1))
                            else:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)))
                    
                    ## ADD BIAS FOR AMORPHOUS VOXELS
                    # if i in countamorph:
                    #     if i == center_id:
                    #         pass
                    #     else:
                    #         prob_matrix[1-x_x, 1-x_y, 1-x_z] = (prob_crys*-1*len(countcrys)*0.1 + (1-prob_matrix[1,1,1]))/len(countamorph) # ((1-prob_self*(1/total_count))-(1/(total_count+len(countcrys))))/len(countamorph)#prob_amorph*(1/total_count)
                    # elif i in countcrys:
                    #     if i == center_id:
                    #         pass
                    #     else:
                    #         prob_matrix[1-x_x, 1-x_y, 1-x_z] = 0.1#((1-prob_self*(1/total_count))-(1/(total_count+len(countamorph))))/len(countcrys)#/(1-len(countcrys))##prob_crys*(1/total_count)
                    # else:
                    #     pass
                    #choice x, choice y, choice z
                    # step_array = np.array([-1, 0, 1])
                    # print(prob_matrix)
                    # print(prob_matrix[1,1,:])
                    # choice_x = np.random.choice(step_array, p=prob_matrix[:,1,1])
                    # choice_y = np.random.choice(step_array, p=prob_matrix[1,:,1])
                    # choice_z = np.random.choice(step_array, p=prob_matrix[1,1,:])
                    # print(choice_x,choice_y,choice_z)
                # print(prob_matrix)
                flat_array = np.arange(np.ndarray.flatten(prob_matrix).shape[0])
                # print(flat_array)
                choice = np.random.choice(flat_array, p=np.ndarray.flatten(prob_matrix))
                flat_array = np.reshape(flat_array, (3,3,3))
                n_p = np.where(flat_array==choice)
                next_pixel = np.vstack((n_p[0], n_p[1], n_p[2])).transpose()[0]
                # print(next_pixel)
                # print(key_matrix)
                next_id = key_matrix[next_pixel[0]][next_pixel[1]][next_pixel[2]]
                # print(flat_array)
                # print(next_id)
                for path_item in path_array:
                    if path_item[0] == j[0]:
                        path_item[t+1] = next_id
        t+=1
        print("t = ", t)

    print(path_array)
    print(path_array.shape)
    # pbar.close()
    return path_array

def iter_comments():
                # print(path_array)
                # print(key_matrix)
                # print(prob_matrix)
                # print(np.sum(prob_matrix))
                    
                    
                    # for j in (x_x, x_y, x_z):
                    #     for k in (x, y, z):
                    #         total_list.append(j+k)
                    #         total_list.append(j-k)
                        # prob_matrix[i-j] = 1/total_count
                        # prob_matrx[i+j] = 1/total_count
                # print(total_list)
        # print(mat_props)
        # for key, value in prop_dict.items():
        # for key in sorted(prop_dict.keys()):
        #     # print(prop_dict[key])
        #     # print(j)
        #     # print(key)
        #     if prop_dict[key][2] == 1:
        #         print(key, prop_dict[key], coords_dict[key])
        #         x,y,z = coords_dict[key]
        #         print(x,y,z)
        #         neighbour_list = neighbours(x,y,z, res=np_array[1])
        #         # print(neighbour_list)
        #         id_list = []
        #         for a in neighbour_list:
        #             x1, y1, z1 = a
        #             print(x1, y1, z1)
        #             if np_array[0][x1, y1, z1] == 1:
        #                 for key1, value1 in sorted(coords_dict.items()):
        #                     # print('value = ', value1)
        #                     if np.all(value1 == [x1, y1, z1]):
        #                         id_list.append(key1)
        #         print(id_list)
        #         #     pass
        # t += 1
        
 

        # t+=1
                    # print(x_x, x_y, x_z)
                    # prob_matrix[]
                    # if (x_x < x and x_y < y and x_z < z):
                    #     prob_matrix[]

                #get id
                # print("probmatrix", prob_matrix)


                        
                        # for index in coords_array:
                        #     if a == coords_array[index][1:3]).all(axis=1))[0]
                        # countamorph.append(a)
                        # for b, c in enumerate(prop_array):
                        #     if c[5] == 1:
                        #         pass
                        # countcrys.append()                    
                                    #match them with coordinates and find out of they are crystalline

                # next_coord = random_walk(k[2])
                # if next_coord is None:
                #     k[2] == 2
                # else:
                #     next_coord = random_walk(j[2])
                
                    
                    # Fick(diff_coeff_mm["37"], i)
    # print(input_dir)
    # print(output_dir)
    return

def iter_path(max_steps, coords_array, prop_array, np_array, id_array, path_array, bias = False):
    #initialise 3D array shape of np_array with zeros that dynamically changes values
    #the values are probability numbers 

    #for each property array row, if voxel is active, run randomwalk
    prob_crys = 0.5
    prob_amorph = 1.5
    prob_self = 0.5
    # print(coords_array)
    # print(prop_array[0])
    coords_dict = {}
    prop_dict = {}
    prop_shape = prop_array[0].shape
    # print(path_array.shape)
    # print(prop_shape)
    for i in coords_array:
        coords_dict[i[0]] = np.array(i[1:4])
    for j in prop_array[0]:
        prop_dict[j[0]] = j[1:prop_shape[1]]
    path_array = np.c_[path_array, np.zeros((path_array.shape[0],max_steps))]
    # print('\n path_array shape = ', path_array.shape)

    for t in tqdm(np.arange(start=1, stop=max_steps+1)):# max_steps+1)):
        # print('\n t = ', t)
        # print(path_array[0])
        # print(path_array[0][t])
        #iterates through coordinate array, instead it should iterate through flowpath array as it needs assign the next coordinate for the path
        for j in path_array:
            if np.all(prop_array[0][:,1]):
                print('all elements pathed')
                break
            # print(j)
            # print(j[t])
            x,y,z = coords_dict[j[t-1]]

            # print(x,y,z)
            #gets coordinates of neighbouring voxels
            neighbour_list = neighbours(x, y, z, res = np_array[1])
            # print(neighbour_list)
            id_list = []
            id_list.append(j[t-1])
            for a in neighbour_list:
                #if pixel is white/polymer
                x1, y1, z1 = a[0], a[1], a[2]
                if np_array[0][x1][y1][z1] == 1: #if pixel is white
                    #get coordinate id
                    id_list.append(id_array[0][x1][y1][z1]) #np.array([x1, y1, z1])
                    id_list = [int(i) for i in id_list]
            #count number of pixels
            countamorph = [] #list containing id of voxels that are neighbouring to the current voxel and are amorphous
            countcrys = [] #same as above, except for crystalline
            #find if voxel is amorphous of crystalline
            for iid in id_list:
                if prop_dict[iid][5] == 1:
                    countcrys.append(iid)
                # if mat_props[0][iid][5] == 1: #if crystalline
                    # countcrys.append(iid)
                else:
                    countamorph.append(iid)
            total_count = len(countcrys) + len(countamorph)
            # print(total_count, len(countamorph), len(countcrys))
            #create a 3D array with current voxel in the middle that is 3x3 and add neighbouring voxels to this array
            prob_matrix = np.zeros(shape=(3,3,3), dtype=float)
            key_matrix = np.zeros(shape=(3,3,3), dtype=int)
            key_matrix[1,1,1] = id_list[0]
            # print(prob_matrix)
            # print(key_matrix)
            prob_matrix[1,1,1] = prob_self*(1/total_count)
            center_id = id_list[0]
            center_coord = coords_dict[center_id]
            center_coord_x = center_coord[0]
            center_coord_y = center_coord[1]
            center_coord_z = center_coord[2]
            # print(id_list[1:])
            for i in id_list[1:]:
                # print(i)
                xx = coords_dict[i]#coords_array[np.where(coords_array[:,0] == i)]
                # print(xx)
                x_x = center_coord_x - xx[0]
                x_y = center_coord_y - xx[1]
                x_z = center_coord_z - xx[2]
                # x_list = [x_x, x_y, x_z]
                key_matrix[1-x_x, 1-x_y, 1-x_z] = i
                # if i == center_id:
                #     prob_matrix[1, 1, 1] = prob_self*(1/total_count)
                if bias == False:
                    prob_matrix[1-x_x, 1-x_y, 1-x_z] = (1-(prob_self/total_count))/(total_count-1)
                        # print(prob_matrix)
                        # print(key_matrix)
                else:
                    if i in countamorph:
                        # if i == center_id:
                        #     pass
                        # else:
                        if prop_dict[i][5] == 0:
                            prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph)-1)
                        else:
                            prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph))
                    elif i in countcrys:
                        # if i == center_id:
                        #     pass
                        # else:
                        if prop_dict[i][5] == 1:
                            prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)-1))
                        else:
                            prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)))
                ## ADD BIAS FOR AMORPHOUS VOXELS
            # print(prob_matrix)
            flat_array = np.arange(np.ndarray.flatten(prob_matrix).shape[0])
            # print(flat_array)
            choice = np.random.choice(flat_array, p=np.ndarray.flatten(prob_matrix))
            flat_array = np.reshape(flat_array, (3,3,3))
            n_p = np.where(flat_array==choice)
            next_pixel = np.vstack((n_p[0], n_p[1], n_p[2])).transpose()[0]
            next_id = key_matrix[next_pixel[0]][next_pixel[1]][next_pixel[2]]
            # print('t+1 = ', t+1)
            # print(j)
            j[t] = next_id
            prop_dict[j[t]][1] = 1
            # print(prop_array[0][:,0])
            # path_id = np.where(prop_array[0][:,0] == j[t])[0]
            # print(j[t])
            prop_array[0][int(j[t])][1] = 1
            # print(path_id)
            # print(prop_dict[j[t]][1])
            # print(j[t])
            # print(prop_array[0][path_id][0][1])
            # prop_array[0][path_id][1] = 1#prop_dict[j[t]][1]
    # print(prop_array[0][0])
            # prop_array[0][prop_array[0][np.where(prop_array[0][:,0] == j[t])][0]] = 
    # print(path_array[0])
    # print(path_array)
    # print(path_array.shape)
    # np.save(output_dir+'patharray.npy', path_array)
    return path_array

def Fick(diff, t, c0 = None, x=1):
    """
    returns C/C0
    C0 = water_conc

    """
    if c0 is None:
        return math.erfc(x/(math.sqrt(4*diff*t)))
    else:
        return c0*math.erfc(x/(math.sqrt(4*diff*t)))

def iter_fick(max_steps, temp, pixel_scale, coords_array, prop_array, np_array, id_array, path_array, bias = False):
    pha_density = 1.240 * (1/1000**2) #g/m3
    diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
    diff = diff_coeff_mm[temp]
    prop_shape = prop_array[0].shape
    voxel_vol = pixel_scale**3 #mm^3
    voxel_mass = pha_density * voxel_vol
    water_mass = voxel_mass * 0.00984
    water_conc = water_mass/voxel_vol
    n = 10
    coords_dict = {i[0]: np.array(i[1:4]) for i in coords_array}
    prop_dict = {j[0]: j[1:prop_shape[1]] for j in prop_array[0]}
    path = iter_path(max_steps, coords_array, prop_array, np_array, id_array, path_array, bias = False)
    # for t in tqdm(np.arange(1, max_steps+1)):
    t=max_steps
    for i in path[0:1]:
        for j,k in enumerate(i):
            avg_conc = Fick(diff_coeff_mm["37"], t, c0=water_conc, x=j)+Fick(diff_coeff_mm["37"], t, c0=water_conc, x=(j+1))
            # print(avg_conc)
            prop_dict[k][4] += avg_conc
            # diff_list = np.zeros(n)
            # conc_list = np.zeros(n-1)
            # for x in np.arange(0, n):
            #     diff_list[x] = Fick(diff_coeff_mm["37"], t, c0=water_conc, x=(j+1/n))
            # for index, x in enumerate(conc_list):
            #     x = (diff_list[index]+diff_list[index+1])/2
            # total_conc = sum(i*1/n for i in conc_list)
            # print(total_conc)
            # prop_dict[k][4] += total_conc
            # # total_conc = np.sum([lambda i: i*(1/n) for i in conc_list])
            # # print(total_conc)
    print(prop_dict[path[0][0]][4])
    return

diff_coeff = {"25": 51.7e-12, "37": 67.6e-12, "50": 165e-12} #x10^(-12) m^2/s
diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
# print(diff_coeff_mm)
pixel_scale = 1 #mm/px * 1px
temp = '37'
# iter_fick(300, temp, pixel_scale, coords, mat_props, nparr, idarr, flowpath)
iter_path(100, coords, mat_props, nparr, idarr, flowpath)
#Fick's Law of Diffusion 
##Assume 1D initially, infinite source
##Molarity (concentration): C = m/V * 1/MW
##m = mass of solute (g), V is volume of solution in (L), MW is molecular weight, C is molar concentration (mol/L)



#Fick's 2nd Law determines concentration change over time - eq. similar to heat eq
#x goes from 0 -> 1 going through the length of the voxel
#iter = time, the function is meant to run each iteration to update the concetration of water in the exposed voxel
#conecntration units: mol/mm^3
#concentration at surface = M_s = saturation mass of water
#therefore, Fick function determines 


    # return (1/(math.sqrt(math.pi*diff*t)))*(math.exp(-((x**2)/(4*diff*t))))

def MolecularWeight(mw0, lmda, t):
    return mw0*math.exp(-lmda*t)

def Modulus(n):
    k_b = 1.38064852e-23 #m^2 kg s^-2 K^-1
    return 3*n*k_b*310.15

#once C/C0 reaches 0.5, random walk takes place
# pha_density = 1.240 * (1/1000**2) #g/m3
# diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
# diff = diff_coeff_mm['37']
# pixel_scale=1
# voxel_vol = pixel_scale**3 #mm^3
# voxel_mass = pha_density * voxel_vol
# water_mass = voxel_mass * 0.00984
# water_conc = water_mass/voxel_vol

# print(Fick(diff_coeff_mm["37"], 30, c0 = water_conc))

#Mw loss function (units g/mol/day) Mw loss rate: 900 g/mol/day
#crystallinity (Poisson distribution)
#relate Mw to mechanical properties
#use voxel model to assign Mw value and reduce it using the loss function
#in the amorphous regions
#maybe nodal FEA modelling
#assumptions:
##each voxel has equal density

def MwLossRate():
    return 900 #g/mol/day

def AssignMw(led):
    if str.lower(led) == "high":
        return 220000
    elif str.lower(led) == "med":
        return 270000
    elif str.lower(led) == "low":
        return 350000
    else:
        return "unknown LED"

def AssignXc(led):
    # def init_crystallinity(self, chi):
    #     # crystallinity = probability a pixel will be crystalline
    #     bin_prob = np.random.binomial(1, chi)
    #     return 1 if bin_prob == 1 else 0
    bin_prob = np.random.binomial(1, chi)
    return 1 if bin_prob == 1 else 0
    