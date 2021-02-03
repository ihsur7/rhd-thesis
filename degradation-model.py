from sys import path
import pydirectory as pyd
import open3d as otd
# from open3d import *
import numpy as np
# import stl
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
from collections import Counter
import vedo

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
            mw0 = np.random.poisson(300000)
        elif led == 0.097:
            mw0 = np.random.poisson(300000)
        elif led == 0.077:
            mw0 = np.random.poisson(250000)
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
        # print('prop array = ', self.prop_array[0])
        for i in self.prop_array:
            if i[3] == 1:
                # print(i)
                patharray.append(i[0])
                i[1] = 1
                # print(i)
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



# print(flowpath)
# prob_array = np.zeros(shape=nparr[1], dtype=float)
# print(nparr[1])
# print(prob_array, prob_array.shape)
# prob_array = np.zeros(shape=(3,3,3), dtype=float)
# print(prob_array)
# print('end', coords)


def iter_path(max_steps, coords_array, prop_array, np_array, id_array, path_array, bias = True):
    #initialise 3D array shape of np_array with zeros that dynamically changes values
    #the values are probability numbers 
    #for each property array row, if voxel is active, run randomwalk
    prob_crys = 0.5
    prob_amorph = 1.5
    #POTENTIALLY ADD PATH SPLIT, use two random number generators between 0 and 1
    #first one tells which path splits
    #second one tells where it splits
    prob_self = 0.001
    pm_p = 0.1
    pm_a = 2
    coords_dict = {}
    prop_dict = {}
    prop_shape = prop_array[0].shape
    coords_dict = {i[0]: np.array(i[1:4]) for i in coords_array}
    prop_dict = {j[0]: j[1:] for j in prop_array[0]}
    bug_check = False
    steps = 1
    while np.all(prop_array[0][:,1]) == False:
        path_array = np.c_[path_array, np.zeros((path_array.shape[0],1))]
        for t in np.arange(start=steps, stop=steps+1):# max_steps+1)):
            # print('\n t = ', t)
            #iterates through coordinate array, instead it should iterate through flowpath array as it needs assign the next coordinate for the path
            for j in path_array:
                if bug_check and np.all(prop_array[0][:, 1]):
                    print('all elements pathed')
                    break
                x,y,z = coords_dict[j[t-1]]
                #gets coordinates of neighbouring voxels
                neighbour_list = neighbours(x, y, z, res = np_array[1])
                id_list = [j[t-1]]
                for a in neighbour_list:
                    #if pixel is white/polymer
                    x1, y1, z1 = a[0], a[1], a[2]
                    if np_array[0][x1][y1][z1] == 1: #if pixel is white
                        #get coordinate id
                        id_list.append(id_array[0][x1][y1][z1]) #np.array([x1, y1, z1])
                        id_list = [int(i) for i in id_list]
                #count number of pixels
                single_px_counter = 0
                if len(id_list) > 1:
                    na = []
                    nc = []
                    npa = []
                    npc = []
                    ns = [id_list[0]]
                    #find if voxel is amorphous of crystalline
                    for iid in id_list[1:]:
                        # print(prop_dict[iid])
                        if prop_dict[iid][4] == 1 and prop_dict[iid][0] == 0:
                            # print(prop_dict[iid])
                            nc.append(iid)
                            # print(nc)
                        elif prop_dict[iid][4] == 1 and prop_dict[iid][0] == 1:
                            npc.append(iid)
                        elif prop_dict[iid][4] == 0 and prop_dict[iid][0] == 0:
                            na.append(iid)
                        else:
                            npa.append(iid)
                    #create a 3D array with current voxel in the middle that is 3x3 and add neighbouring voxels to this array
                    totaln = len(na) + len(nc) + len(npa) + len(npc) + 1 #1 for self
                    prob_matrix = np.zeros(shape=(3,3,3), dtype=float)
                    key_matrix = np.zeros(shape=(3,3,3), dtype=int)
                    key_matrix[1,1,1] = id_list[0]
                    sum_prob = ((totaln-1)*(totaln-prob_self))/totaln
                    npa_na = (pm_p*len(npa))+len(na)
                    npc_nc = (pm_p*len(npc))+len(nc)
                    # print('npa_na: ', npa_na, '\nnpa_nc: ', npc_nc)
                    # if npa_na == 0 and npc_nc == 0:
                    #     print('id_list contains no amorphous and crystalline voxels: ', id_list)
                    #     print('na: ', na, '\nnc: ', nc, '\nnpa: ', npa, '\nnpc: ', npc)
                    #     print('npa_na: ', npa_na, 'npc_nc: ', npc_nc)
                    #     print('sum_prob: ', sum_prob)                        
                    #     print('neighbours: ', neighbour_list)
                    #     for i in id_list:
                    #         print(prop_dict[i])
                    # print(prop_dict[199])
                    pm_c = sum_prob/((pm_a*npa_na)+npc_nc)
                    pm_a = pm_a*pm_c
                    ppa = pm_a*pm_p
                    ppc = pm_c*pm_p
                    center_id = id_list[0]
                    center_coord = coords_dict[center_id]
                    center_coord_x = center_coord[0]
                    center_coord_y = center_coord[1]
                    center_coord_z = center_coord[2]
                    for i in id_list[1:]:
                        xx = coords_dict[i]
                        x_x = center_coord_x - xx[0]
                        x_y = center_coord_y - xx[1]
                        x_z = center_coord_z - xx[2]
                        # x_list = [x_x, x_y, x_z]
                        key_matrix[1-x_x, 1-x_y, 1-x_z] = i
                        if bias == False:
                            prob_matrix[1-x_x, 1-x_y, 1-x_z] = (1-(prob_self/totaln))/(totaln-1)
                        else:
                            prob_matrix[1,1,1] = prob_self/totaln
                            if i in na:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = pm_a/(totaln-1)
                            elif i in nc:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = pm_c/(totaln-1)
                            elif i in npa:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = ppa/(totaln-1)
                            else:
                                prob_matrix[1-x_x, 1-x_y, 1-x_z] = ppc/(totaln-1)
                    flat_array = np.arange(np.ndarray.flatten(prob_matrix).shape[0])
                    choice = np.random.choice(flat_array, p=np.ndarray.flatten(prob_matrix))
                    flat_array = np.reshape(flat_array, (3,3,3))
                    n_p = np.where(flat_array==choice)
                    next_pixel = np.vstack((n_p[0], n_p[1], n_p[2])).transpose()[0]
                    next_id = key_matrix[next_pixel[0]][next_pixel[1]][next_pixel[2]]
                    j[t] = next_id
                    prop_dict[j[t]][1] = 1
                    prop_array[0][int(j[t])][1] = 1
                else:
                    single_px_counter += 1
            # np.save(output_dir+'patharray.npy', path_array)
            steps += 1
            # print(steps)
    else:
        print('All voxels pathed.')
        print('Steps required: ', steps)
    print('#single pixels: ', single_px_counter)

    # for i in dist_array:
    #     dupe_list = [k for k,v in Counter(i).items() if v>1]

    # print(dupe_list)
    np.save(output_dir+'path_array', np.asarray_chkfinite((path_array, prop_array, prop_dict, coords_dict)), allow_pickle=True)
    print('paths saved...')
    # np.save(output_dir+'prop_array', prop_array)
    return path_array, prop_array, prop_dict, coords_dict

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
    prop_dict = {j[0]: j[1:] for j in prop_array[0]}
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

def Fick(diff, t, c0 = None, x=1):
    """
    returns C/C0
    C0 = water_conc

    """
    if c0 is not None:
        return c0*math.erfc(x/(math.sqrt(4*diff*t)))

    conc_ratio = math.erfc(x/(math.sqrt(4*diff*t)))
    # print(conc_ratio)
    if conc_ratio < 1:
        return conc_ratio
    else:
        return 1

def loss_rate_calc(loss_rate, conc_ratio, type='linear'):
    if conc_ratio == 0:
        return 0
    else:
        return loss_rate*(conc_ratio)

def MwLoss(mw0, c, c0, avg_loss_rate, t):
    # print(c/c0)
    grad1 = grad_calc(avg_loss_rate, conc_ratio=c/c0)
    loss_rate = grad1*avg_loss_rate
    # print(loss_rate)
    return mw0*math.e**(-loss_rate*t)

def MwLossData(temp, path, time_array, gradtype='linear'):
    # loss_rate= [0.001776, 0.002112, 0.002527] #ln, in weeks too high, may need to be in seconds
    loss_rate = [4.900e-008, 4.737e-008, 4.262e-008] #ln, in seconds
    # loss_rate = [0.0007610, 0.0008171, 0.001194]
    # loss_rate = [0.0002538, 0.0003017, 0.0003611] #ln, in days    
    average_loss_rate = np.average(loss_rate)
    print("Avg. Loss Rate: ", average_loss_rate)
    pha_density = 0.00124 #g/mm3 #1.240 * (1/1000**2) #g/m3
    diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
    diff = diff_coeff_mm[temp]
    voxel_vol = pixel_scale**3 #mm^3
    voxel_mass = pha_density * voxel_vol
    water_mass = voxel_mass * 0.00984
    mw_water = 18.01528 #g/mol
    mw_pbs = 411.04 #g/mol
    water_conc = (water_mass/voxel_vol)*(1/mw_water)
    print("C0: ", water_conc)
    # print(path[1])
    data_array = [i[0] for i in path[1][0]]
    data_array = np.asarray_chkfinite(data_array)
    data_array = data_array.reshape(data_array.shape[0],-1)
    conc_array = data_array
    # data_array = np.empty(shape=(path[1][0].shape[0],len(time_array+1)))
    # data_dict = {i: [path[2][i][5]] for i in path[2]}
    # data_dict = {j[0]: j[5] for j in path[2][j]}
    # print(data_dict)
    data_array = np.c_[data_array, np.zeros(shape=(data_array.shape[0], time_array.shape[0]))]
    # for i in data_array:
        # if i[0] == 123:
            # print('123: ', i)
        # i[1] = path[2][i[0]][5]
    # print(data_array.shape)
    conc_array = np.c_[conc_array, np.zeros(shape=(conc_array.shape[0], time_array.shape[0]))]
    # print(conc_array.shape)
    for i in data_array:
        i[1] = path[2][i[0]][5]
    for j in conc_array:
        j[1] = path[2][j[0]][3]
    data_dict = {i[0]: i[1:] for i in data_array}
    conc_dict = {j[0]: j[1:] for j in conc_array}
    # print(len(conc_dict[9]))
    # print(path[1][0][123], path[2][123])
    # print(data_dict)
    # print(data_dict[123])
    # print(data_array.shape, path[1][0].shape)   
    # print([i*604800 for i in time_array]) 
    # print(path[0][0])

    for tindex, t in enumerate(time_array):
        for p in path[0]:
            for index, q in enumerate(p):
                tt = t*604800 #weeks in seconds
                if t == 0:
                    avg_conc_ratio = 0
                else:
                    # print((index+1)*pixel_scale)
                    avg_conc_ratio = (Fick(diff_coeff_mm["37"], tt, c0=None, x=index*pixel_scale) + Fick(diff_coeff_mm["37"], tt, c0=None, x=(index+1)*pixel_scale))/2

                avg_conc = avg_conc_ratio*water_conc
                # if q == 9:
                #     print(avg_conc_ratio, avg_conc)
                #     print(conc_dict[9])
                conc_dict[q][tindex] = avg_conc_ratio
                conc_array[index][tindex+1] = avg_conc_ratio

                path[2][q][3] = avg_conc
                path[1][0][index+1][4] = avg_conc
                # print(path[2][q][5])
                if gradtype = "linear":
                    grad = avg_conc_ratio*average_loss_rate
                    # grad = loss_rate_calc(average_loss_rate, avg_conc_ratio)
                elif gradtype = "exp":
                    grad = 
                mwt = path[2][q][5]*math.e**(-1*(grad)*tt)
                # mwt = MwLoss(path[2][q][5], avg_conc, water_conc, average_loss_rate, tt)
                # print(tindex)
                data_dict[q][tindex] = mwt
                data_array[index][tindex+1] = mwt
                # print(data_dict[q], data_dict[q][tindex])
                # print(data_array[index], data_array[index][tindex+1])

    return path, data_array, data_dict

def AssignMw(led):
    if str.lower(led) == "high":
        return 220000
    elif str.lower(led) == "med":
        return 270000
    elif str.lower(led) == "low":
        return 350000
    else:
        return "unknown LED"

def MolecularWeight(mw0, lmda, t):
    return mw0*math.exp(-lmda*t)

def Modulus(n):
    k_b = 1.38064852e-23 #m^2 kg s^-2 K^-1
    return 3*n*k_b*310.15

def AssignXc(led):
    # def init_crystallinity(self, chi):
    #     # crystallinity = probability a pixel will be crystalline
    #     bin_prob = np.random.binomial(1, chi)
    #     return 1 if bin_prob == 1 else 0
    bin_prob = np.random.binomial(1, chi)
    return 1 if bin_prob == 1 else 0
    

if __name__ == "__main__":
    print('numpy version = ', np.__version__)
    # print('open3d version: = ', otd.__version__)

    #Set up directories
    in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"
    out_dir = "/data/deg_test_output/"
    dir = pyd.Directory(in_dir, out_dir)
    input_dir = dir.InputDIR()
    output_dir = dir.OutputDIR()
    new_input_dir = pyd.Directory(out_dir).InputDIR()
    # print(os.path.exists(output_dir+'path_array.npy'))
    if os.path.exists(output_dir+'path_array.npy'):
        print('file found...')
    print('checking if file contents match input data...')
    path = np.load(output_dir+'path_array.npy',allow_pickle=True)
    num_adj = np.where(path[0][:,3] == 1)[0].shape

    if path[0].shape[0] != num_adj[0]:
        print('file does not match... creating new data')
        a = Voxelize(input_dir)
        coords = a.coord_array()
        props = a.matprops_array(numprops=8)
        nparr = a.to_numpy()
        idarr = a.vox_id()
        mat_props = InitPixelClassifier(coords, nparr, props).init_classify(0.097, 0.67, 3) 
        flowpath = PathArray(coords, nparr, mat_props[0]).initPathArray()
        path = iter_path(2, coords, mat_props, nparr, idarr, flowpath, bias=True)
    else:
        print('file matches input data')
    #Set up model and metadata

    print('model resolution = ', nparr[0].shape)
    print('# polymer voxels = ', coords.shape)
    #25 = 211.88 seconds on macbook, ~50k paths (flow vectors)
    #50 = 
    # print(mat_props[0])
    print('# adjacent = ', num_adj)

    time_array = np.arange(start=0, stop=21, step=1)
    print('# timepoints: ', time_array.shape[0], '\ntimepoints: ', time_array)
    
    diff_coeff = {"25": 51.7e-12, "37": 67.6e-12, "50": 165e-12} #x10^(-12) m^2/s
    diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
    # print(diff_coeff_mm)
    pixel_scale = 0.25/35 #mm/px * 1px
    temp = '37'
    # iter_fick(300, temp, pixel_scale, coords, mat_props, nparr, idarr, flowpath)

    #Calculate Flowpath


    
    # print('# paths = ', flowpath.shape)

    #Calculate Molecular Weight
    mw_data = MwLossData("37", path, time_array, gradtype='exp')


    # print(idarr[0].shape)
    vol = vedo.Volume(idarr[0])
    vol.addScalarBar3D()

    mw_data_array = idarr[0]

    # for i in mw_data_array:
    #     for j in i:
    #         for k in j:
    #             k = mw_data[1][k][1]
        

    lego = vol.legosurface(vmin=1, vmax=coords.shape[0])
    lego.addScalarBar3D()
    text1 = vedo.Text2D('Make a Volume from numpy.mgrid', c='blue')
    text2 = vedo.Text2D('its lego isosurface representation\nvmin=1, vmax=2', c='dr')
    # print(mw_data[1][123])
    
    avg_mw = [np.average(mw_data[1][:,i]) for i in np.arange(1, time_array.shape[0]+1)]
    print("Avg Mw: ", avg_mw)
    save_array = np.zeros(shape=(time_array.shape[0], 2))
    print(save_array.shape)
    # print(save_array)
    for index, i in enumerate(time_array):
        save_array[index][0] = i
        save_array[index][1] = avg_mw[index]
    pd.DataFrame(save_array).to_csv(output_dir+'mw_data.csv')
    # np.savetxt(out_dir+'mw_data', save_array, allow_pickle=True)
    # print('numpy array from Volume:', 
    #     vol.getPointArray().shape, 
    #     vol.getDataArray().shape)

    # vedo.show([(vol,text1), (lego,text2)], N=2, azimuth=10)
    # vedo.show(lego,axes=1)
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

# def iter(coords_array, prop_array, np_array, id_array, path_array, bias = False):
#     #initialise 3D array shape of np_array with zeros that dynamically changes values
#     #the values are probability numbers 
#     # prob_array = np.zeros(shape=np_array[1], dtype=float)
#     # prob_array = np.zeros(shape=(3,3,3), dtype=float)
#     # print(prob_array)

#     #for each property array row, if voxel is active, run randomwalk
#     t = 0
#     print("t = ", t)

#     max_steps = 3
#     prob_crys = 0.5
#     prob_amorph = 1.5
#     prob_self = 0.5
#     # print(coords_array)
#     # print(prop_array[0])
#     coords_dict = {}
#     prop_dict = {}
#     prop_shape = prop_array[0].shape
#     # print(path_array.shape)
#     # print(prop_shape)
#     for i in coords_array:
#         coords_dict[i[0]] = np.array(i[1:4])
#     for j in prop_array[0]:
#         prop_dict[j[0]] = j[1:prop_shape[1]]
#     # print(prop_dict)
#     # print(mat_props)
#     # print(path_array)
#     # print(coords_dict[0])
#     # for t in tqdm(range(max_steps)):
#     # pbar = tqdm(total=max_steps+1)
#     while t < max_steps:
#         path_array = np.c_[path_array, np.zeros(path_array.shape[0])]
#         it = 0
#         #iterates through coordinate array, instead it should iterate through flowpath array as it needs assign the next coordinate for the path
#         for j in coords_array:
#             # print(t)
#             # print(path_array.shape)
#             # print(j)
#             #checks if voxel is active, find neighbouring voxels
#             if prop_dict[j[0]][2] == 1:
#                 # print(j)
#                 it += 1
#             # print(k)
#             # if k[2] == 1: 
#                 # print(coords_array)
#                 x,y,z = coords_dict[j[0]]
#                 # x, y, z = coords_array[j][1], coords_array[j][2], coords_array[j][3]
#                 #gets coordinates of neighbouring voxels
#                 neighbour_list = neighbours(x, y, z, res = np_array[1])
#                 # print(neighbour_list)
#                 id_list = []
#                 id_list.append(j[0])
#                 for a in neighbour_list:
#                     #if pixel is white/polymer
#                     # print(a)
#                     x1, y1, z1 = a[0], a[1], a[2]
#                     # print(np_array[0].shape)
#                     # print(np_array[0][x1][y1][z1])
#                     # print(x1, y1, z1)
#                     if z1 == 56:
#                         print(neighbour_list)
#                         print(a)
#                         print(z1)
#                     # print(np_array[0][:,:,56])
                    
#                     # print('shape = ', np_array[0].shape)
#                     if np_array[0][x1][y1][z1] == 1: #if pixel is white
#                         #get coordinate id
                        
#                         id_list.append(id_array[0][x1][y1][z1]) #np.array([x1, y1, z1])
#                         # print(id_list)
#                         id_list = [int(i) for i in id_list]
#                         # loc = np.where((coords_array[:,1:4] == coord).all(axis=1))
#                         # print(loc[0])
#                         # for ii in loc[0]:
#                         #     id_list.append(coords_array[ii,0])
#                         # print("idlist = ",id_list)
#                         # loc = np.where((coords_array[1:3] == np.array(a)).all(axis=1))
#                         #count number of pixels
#                 # print('it = ', it)
#                 # print('idlist = ', id_list)
#                 countamorph = [] #list containing id of voxels that are neighbouring to the current voxel and are amorphous
#                 countcrys = [] #same as above, except for crystalline
#                 #find if voxel is amorphous of crystalline
#                 # print(mat_props[0][1][5])
#                 for iid in id_list:
#                     if prop_dict[iid][5] == 1:
#                         countcrys.append(iid)
#                     # if mat_props[0][iid][5] == 1: #if crystalline
#                         # countcrys.append(iid)
#                     else:
#                         countamorph.append(iid)
#                 total_count = len(countcrys) + len(countamorph)
#                 # print(total_count, len(countamorph), len(countcrys))
#                 #create a 3D array with current voxel in the middle that is 3x3 and add neighbouring voxels to this array
#                 prob_matrix = np.zeros(shape=(3,3,3), dtype=float)
#                 key_matrix = np.zeros(shape=(3,3,3), dtype=int)
#                 key_matrix[1,1,1] = id_list[0]
#                 # print(prob_matrix)
#                 # print(key_matrix)
#                 prob_matrix[1,1,1] = prob_self*(1/total_count)
#                 center_id = id_list[0]
#                 center_coord = coords_dict[center_id]
#                 center_coord_x = center_coord[0]
#                 center_coord_y = center_coord[1]
#                 center_coord_z = center_coord[2]
#                 # print(id_list[1:])
#                 for i in id_list[1:]:
#                     # print(i)
#                     xx = coords_dict[i]#coords_array[np.where(coords_array[:,0] == i)]
#                     # print(xx)
#                     x_x = center_coord_x - xx[0]
#                     x_y = center_coord_y - xx[1]
#                     x_z = center_coord_z - xx[2]
#                     # x_list = [x_x, x_y, x_z]
#                     key_matrix[1-x_x, 1-x_y, 1-x_z] = i
#                     # if i == center_id:
#                     #     prob_matrix[1, 1, 1] = prob_self*(1/total_count)
#                     if bias == False:
#                         prob_matrix[1-x_x, 1-x_y, 1-x_z] = (1-(prob_self/total_count))/(total_count-1)
#                             # print(prob_matrix)
#                             # print(key_matrix)
#                     else:
#                         if i in countamorph:
#                             # if i == center_id:
#                             #     pass
#                             # else:
#                             if prop_dict[i][5] == 0:
#                                 prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph)-1)
#                             else:
#                                 prob_matrix[1-x_x, 1-x_y, 1-x_z] = (((prob_self/total_count))/(total_count-1))*prob_amorph*(len(countamorph))
#                         elif i in countcrys:
#                             # if i == center_id:
#                             #     pass
#                             # else:
#                             if prop_dict[i][5] == 1:
#                                 prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)-1))
#                             else:
#                                 prob_matrix[1-x_x, 1-x_y, 1-x_z] = ((1-(prob_self/total_count))/(total_count-1)*prob_crys*(len(countcrys)))
                    
#                     ## ADD BIAS FOR AMORPHOUS VOXELS
#                     # if i in countamorph:
#                     #     if i == center_id:
#                     #         pass
#                     #     else:
#                     #         prob_matrix[1-x_x, 1-x_y, 1-x_z] = (prob_crys*-1*len(countcrys)*0.1 + (1-prob_matrix[1,1,1]))/len(countamorph) # ((1-prob_self*(1/total_count))-(1/(total_count+len(countcrys))))/len(countamorph)#prob_amorph*(1/total_count)
#                     # elif i in countcrys:
#                     #     if i == center_id:
#                     #         pass
#                     #     else:
#                     #         prob_matrix[1-x_x, 1-x_y, 1-x_z] = 0.1#((1-prob_self*(1/total_count))-(1/(total_count+len(countamorph))))/len(countcrys)#/(1-len(countcrys))##prob_crys*(1/total_count)
#                     # else:
#                     #     pass
#                     #choice x, choice y, choice z
#                     # step_array = np.array([-1, 0, 1])
#                     # print(prob_matrix)
#                     # print(prob_matrix[1,1,:])
#                     # choice_x = np.random.choice(step_array, p=prob_matrix[:,1,1])
#                     # choice_y = np.random.choice(step_array, p=prob_matrix[1,:,1])
#                     # choice_z = np.random.choice(step_array, p=prob_matrix[1,1,:])
#                     # print(choice_x,choice_y,choice_z)
#                 # print(prob_matrix)
#                 flat_array = np.arange(np.ndarray.flatten(prob_matrix).shape[0])
#                 # print(flat_array)
#                 choice = np.random.choice(flat_array, p=np.ndarray.flatten(prob_matrix))
#                 flat_array = np.reshape(flat_array, (3,3,3))
#                 n_p = np.where(flat_array==choice)
#                 next_pixel = np.vstack((n_p[0], n_p[1], n_p[2])).transpose()[0]
#                 # print(next_pixel)
#                 # print(key_matrix)
#                 next_id = key_matrix[next_pixel[0]][next_pixel[1]][next_pixel[2]]
#                 # print(flat_array)
#                 # print(next_id)
#                 for path_item in path_array:
#                     if path_item[0] == j[0]:
#                         path_item[t+1] = next_id
#         t+=1
#         print("t = ", t)

#     print(path_array)
#     print(path_array.shape)
#     # pbar.close()
#     return path_array

# def iter_comments():
#                 # print(path_array)
#                 # print(key_matrix)
#                 # print(prob_matrix)
#                 # print(np.sum(prob_matrix))
                    
                    
#                     # for j in (x_x, x_y, x_z):
#                     #     for k in (x, y, z):
#                     #         total_list.append(j+k)
#                     #         total_list.append(j-k)
#                         # prob_matrix[i-j] = 1/total_count
#                         # prob_matrx[i+j] = 1/total_count
#                 # print(total_list)
#         # print(mat_props)
#         # for key, value in prop_dict.items():
#         # for key in sorted(prop_dict.keys()):
#         #     # print(prop_dict[key])
#         #     # print(j)
#         #     # print(key)
#         #     if prop_dict[key][2] == 1:
#         #         print(key, prop_dict[key], coords_dict[key])
#         #         x,y,z = coords_dict[key]
#         #         print(x,y,z)
#         #         neighbour_list = neighbours(x,y,z, res=np_array[1])
#         #         # print(neighbour_list)
#         #         id_list = []
#         #         for a in neighbour_list:
#         #             x1, y1, z1 = a
#         #             print(x1, y1, z1)
#         #             if np_array[0][x1, y1, z1] == 1:
#         #                 for key1, value1 in sorted(coords_dict.items()):
#         #                     # print('value = ', value1)
#         #                     if np.all(value1 == [x1, y1, z1]):
#         #                         id_list.append(key1)
#         #         print(id_list)
#         #         #     pass
#         # t += 1
        
 

#         # t+=1
#                     # print(x_x, x_y, x_z)
#                     # prob_matrix[]
#                     # if (x_x < x and x_y < y and x_z < z):
#                     #     prob_matrix[]

#                 #get id
#                 # print("probmatrix", prob_matrix)


                        
#                         # for index in coords_array:
#                         #     if a == coords_array[index][1:3]).all(axis=1))[0]
#                         # countamorph.append(a)
#                         # for b, c in enumerate(prop_array):
#                         #     if c[5] == 1:
#                         #         pass
#                         # countcrys.append()                    
#                                     #match them with coordinates and find out of they are crystalline

#                 # next_coord = random_walk(k[2])
#                 # if next_coord is None:
#                 #     k[2] == 2
#                 # else:
#                 #     next_coord = random_walk(j[2])
                
                    
#                     # Fick(diff_coeff_mm["37"], i)
#     # print(input_dir)
#     # print(output_dir)
#     return



    # for j in path[0]:
    #     for k,l in enumerate(j):
    #         # for m, n in enumerate(time_array):
    #             # t = n*604800

    #             # data_dict[l][m+1] = 
    #         for i,tt in enumerate(time_array):
    #             t = tt*604800
    #             #max ratio = 1
    #             if tt == 0:
    #                 avg_conc_ratio = 0
    #             else:
    #                 avg_conc_ratio = (Fick(diff_coeff_mm["37"], t, c0=None, x=k)+Fick(diff_coeff_mm["37"], tt, c0=None, x=(k+1)))/2
    #             avg_conc = avg_conc_ratio*water_conc
    #             conc_dict[l][i+1] = avg_conc
    #             conc_array[k][i+1] = avg_conc
    #             print(i, tt, conc_dict[123][tt])
    #             # if tt == 20:
    #                 # print(conc_dict[123], '\n', conc_array[123]])
    #             # avg_conc = water_conc * avg_conc
    #             # if l == 123:
    #             #     print(avg_conc_ratio)
    #             # print(avg_conc)
    #             # print('conc: ', path[2][l])
    #             path[2][l][3] = avg_conc #dictionary (used to be +=)
    #             path[1][0][k][4] = avg_conc #array
    #             mwt = MwLoss(path[2][l][5], avg_conc, water_conc, average_loss_rate, t)
    #             data_dict[l][i+1] = mwt
    #             data_array[k][i+1] = mwt
                # data_dict[l].append(mwt)
                # if l == 123:
                    # print(data_dict[123])
    # for i in tqdm(time_array):
    #     t = i*604800
    #     print('timepoint: ', i)
    #     # print('\ndictionary: ', path[2][0])
    #     # print('\narray: ', path[1][0][0])
    #     for j in path[0]:
    #         for k,l in enumerate(j):
    #             avg_conc = Fick(diff_coeff_mm["37"], t, c0=water_conc, x=k)+Fick(diff_coeff_mm["37"], t, c0=water_conc, x=(k+1))
    #             # print(avg_conc)
    #             # print('conc: ', path[2][l])
    #             path[2][l][3] += avg_conc #dictionary
    #             path[1][0][k][4] += avg_conc #array
    #             mwt = MwLoss(path[2][l][5], avg_conc, water_conc, average_loss_rate, t)
    #             data_dict[l].append(mwt)
    #             if l == 123:
    #                 print(data_dict[123])
    #             # path[2][l][5] = mwt
    #             # path[1][0][k][6] = mwt
    #             # Fick(diff, i*604800, c0=water_conc, x=path[2])
    # # print(data_dict[123])
    # print(len(data_dict[123]), len(time_array))
    # print('concentration: ', conc_dict[9])
    # print(len(conc_dict[9]))
    # print('mw data: ', data_dict[9])
    #create a 4D array with t 3D array containing mw data
    # print(data_array[123])