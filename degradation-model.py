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
import math
import uuid

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
        [id, nth pixel, pixel state, adjacent, water diffusion, chi, molecular weight, e]
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

    def init_classify(self, led, chi, e):
        '''
        [id, nth pixel, pixel state, adjacent, water diffusion, is crystalline, molecular weight, e]

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
            j[0] = self.init_id()
            j[1] = self.nth_pixel()
            j[2] = self.init_pixel_state()
            j[3] = adjacent(i, self.coords_array, loc_arr)[0]
            j[4] = self.init_diffusion()
            j[5] = self.init_crystallinity(chi)
            j[6] = self.init_molecular_weight(led)
            j[7] = self.init_modulus(e)
            # print(i)
            # j[3] = self.init_adjacent(i, loc_arr)
        return self.prop_array, loc_arr
    
    def init_id(self):#, index):
        # return [i for i in self.coords_array[index]]
        return uuid.uuid4().hex
    
    def nth_pixel(self):
        return 0

    def init_diffusion(self):
        return False

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
        a = np.where((loc_arr == self.coords_array[index]).all(axis=1))[0]
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

def random_walk(isactive=True):
    step_set = [-1, 0, 1]
    dims = 3
    # i = 0
    while isactive == True:
        next_step = np.random.choice(a=step_set, size=(1, dims))
    return next_step

a = Voxelize(input_dir)
coords = a.coord_array()
props = a.matprops_array(numprops=8)
nparr = a.np_array()
mat_props = InitPixelClassifier(coords, nparr, props)

def iter():
    if props[3] == True
    
# print(input_dir)
# print(output_dir)

#Fick's Law of Diffusion 
##Assume 1D initially, infinite source
##Molarity (concentration): C = m/V * 1/MW
##m = mass of solute (g), V is volume of solution in (L), MW is molecular weight, C is molar concentration (mol/L)

diff_coeff = {"25": 51.7e-12, "37": 67.6e-12, "50": 165e-12} #x10^(-12) m^2/s
diff_coeff_mm = {"37": 67.6e-5} #mm^2/s
# print(diff_coeff_mm)
pha_density = 1.240 * (1/1000**2) #g/m3
pixel_scale = 1 #mm/px * 1px
voxel_vol = 1 #mm^3
voxel_mass = pha_density * voxel_vol
water_mass = voxel_mass * 0.00984
water_conc = water_mass/voxel_vol

#Fick's 2nd Law determines concentration change over time - eq. similar to heat eq
#x goes from 0 -> 1 going through the length of the voxel
#iter = time, the function is meant to run each iteration to update the concetration of water in the exposed voxel
#conecntration units: mol/mm^3
#concentration at surface = M_s = saturation mass of water
#therefore, Fick function determines 

def Fick(diff, x, t):
    """
    returns C/C0
    C0 = water_conc
    """
    return math.erfc(x/(math.sqrt(4*diff*t)))
    # return (1/(math.sqrt(math.pi*diff*t)))*(math.exp(-((x**2)/(4*diff*t))))

#once C/C0 reaches 0.5, random walk takes place (initially model 1D flow, move water concentration to next pixel)


print(Fick(diff_coeff_mm["37"], 1, 1e30))

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
    