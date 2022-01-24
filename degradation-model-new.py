import os
import pydirectory as pyd
import numpy as np
import pandas as pd
import PIL.Image as Image
import random
import scipy.ndimage as ndimage
import pyvista
import PVGeo as pvgeo
from tqdm import tqdm

class Presets:
    def __init__(self, led):
        self.led = led
        self.preset = {0.135: {"mw_loss_coeff": 1369}, \
                    0.097: {"mw_loss_coeff": 1220}, \
                    0.077: {"mw_loss_coeff": 1034}}

    def mw_loss_coeff(self):
        return self.preset[self.led]["mw_loss_coeff"]
    
class Voxelize_DF:
    def __init__(self,directory,booltype=False):
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
        for i, j in enumerate(os.listdir(self.directory)):
            arr[:, :, i] = np.asarray(Image.open(self.directory + j), dtype=bool)
        return arr, res1

    def main_df(self):
        arr = self.to_numpy()[0]
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]),3), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i][0:3] = coord[:,i]
        coords_df = pd.DataFrame(coords,columns=['x', 'y', 'z'])
        props = np.ndarray((len(coord[0]),7))
        props_df = pd.DataFrame(props,columns = ['is_pathed?', 'pixel_state', 'adjacent', \
            'water_conc', 'is_crys', 'mw', 'e'])
        main_df = pd.merge(coords_df, props_df,left_index=True, right_index=True)
        # main_df = pd.concat([coords_df,props_df])
        # main_df = pd.DataFrame(np.hstack([coords_df, props_df]))
        return main_df

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


def adjacent(index, main_df, loc_df):
    """
    1 = adjacent
    0 = far
    """
    # print(main_df.iloc[index, 0:3])

    a = np.where((loc_df == main_df.iloc[index, 0:3]).all(axis=1))[0]
    return (1, a) if a.size else (0, a)

class InitPixelClassifier:
    """Object Class that initialises the properties of the voxel model.
    """

    def __init__(self, main_df, np_array):
        self.main_df = main_df
        self.np_array = np_array

    def init_classify(self, led, chi, e):
        '''
        [is_pathed?, pixel state, adjacent, water concentration, is crystalline, molecular weight, e]

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
        print('shape = ', self.np_array.shape)
        loc_arr = np.vstack((loc[0], loc[1], loc[2])).transpose()
        loc_df = pd.DataFrame(loc_arr, columns=['x','y','z'])
        for i in self.main_df.index:
            self.main_df.at[i,'pixel_state'] = self.init_pixel_state()
            self.main_df.at[i,'adjacent'] = adjacent(i, self.main_df, loc_df)[0]
            self.main_df.at[i,'is_crys'] = self.init_crystallinity(chi)
            self.main_df.at[i,'mw'] = self.init_molecular_weight(led)
            self.main_df.at[i,'e'] = self.init_modulus(e)

        return self.main_df, loc_df

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
        return 1
        
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

if __name__ == "__main__":
    in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"
    out_dir = "/data/deg_test_output/"
    dir = pyd.Directory(in_dir, out_dir)
    input_dir = dir.InputDIR()
    output_dir = dir.OutputDIR()
    new_input_dir = pyd.Directory(out_dir).InputDIR()
    im = Voxelize_DF(input_dir)

    im_class = InitPixelClassifier(im.main_df(), im.to_numpy()[0]).init_classify(0.097, 0.67, 3)