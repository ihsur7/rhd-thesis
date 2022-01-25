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

    def coord_array(self):
        arr = self.to_numpy()[0]
        coord = np.asarray(np.where(arr))
        coords = np.empty((len(coord[0]),3), dtype=np.int64)
        for i in np.arange(len(coord[0])):
            coords[i][0:3] = coord[:,i]
        return coords, coord

    def main_df(self):
        coords_df = pd.DataFrame(self.coord_array()[0],columns=['x', 'y', 'z'])
        props = np.ndarray((len(self.coord_array()[1][0]),7))
        props_df = pd.DataFrame(props,columns = ['is_pathed?', 'pixel_state', 'adjacent', \
            'water_conc', 'is_crys', 'mw', 'e'])
        main_df = pd.merge(coords_df, props_df,left_index=True, right_index=True)
        # main_df = pd.concat([coords_df,props_df])
        # main_df = pd.DataFrame(np.hstack([coords_df, props_df]))
        return main_df

    def vox_id(self):
        arr1 = self.to_numpy()[0]
        arr = np.copy(arr1)
        # print(arr)
        main_df = self.main_df()
        for i in main_df.index:
            # print(i)
            x = main_df.at[i, 'x']
            y = main_df.at[i, 'y']
            z = main_df.at[i, 'z']
            # print(x,y,z)
            # x,y,z = main_df.iloc[i, 0:3]
            arr[x,y,z] = i
        return arr

        
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
                ).all():
                    n_list.append([int(i), int(j), int(k)])
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

class PathArray:
    def __init__(self, main_df, np_array):
        self.main_df = main_df
        self.np_array = np_array

    def initPathArray(self):
        """
        Returns:
            array of (n,1) dimensions containing ID of voxels starting with the first exposed voxel
        """
        patharray = [] #np.zeros(shape=(np.count_nonzero(self.prop_array[:,3]), 1), dtype=int)

        for i in self.main_df.index:
            if self.main_df.at[i, 'adjacent'] == 1:
                patharray.append(i)
                self.main_df.at[i, 'is_pathed?'] = 1
        patharray = np.asarray(patharray)
        path_df = pd.DataFrame(patharray.reshape(-1, len(patharray)))
        # print(path_df)
        # print('parray shape = ', patharray.shape)
        # patharray = patharray.reshape(patharray.shape[0],-1)
        return path_df

def iter_path(max_steps, main_df, np_array, path_df, id_array, res, bias = True):
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
    # coords_dict = {i: np.array(main_df.iloc[i, 0:3]) for i in main_df.index}
    # prop_dict = {i: np.array(main_df.iloc[i, 3:]) for i in main_df.index}
    # print(coords_dict,prop_dict)
    bug_check = False
    steps = 1
    print(np.all(main_df['is_pathed?']))
    while np.all(main_df['is_pathed?']) == False:
        # path_array = np.vstack([path_array, np.zeros((path_array.shape[0], steps+1))])
        # path_array = np.c_[path_array, np.zeros((path_array.shape[0],1))]
        path_df = path_df.append(pd.DataFrame(np.zeros((steps+1, path_df.shape[1]), dtype=np.int64)))
        path_df = path_df.reset_index(drop=True)
        for t in tqdm(np.arange(start=steps, stop=steps+1)):# max_steps+1)):
            #iterates through coordinate array, instead it should iterate through flowpath array as it needs assign the next coordinate for the path
            for j in path_df:
                if bug_check and np.all(main_df['is_pathed?']):
                    print('all elements pathed')
                    break
                print(path_df)
                print(main_df.iloc[[path_df[j][t-1]]])
                x,y,z = main_df.iloc[[path_df[j][t-1]]]
                print('\n ', x,y,z)
                #gets coordinates of neighbouring voxels
                neighbour_list = neighbours(x, y, z, res = res)
                id_list = [path_df[j][t-1]]
                for a in neighbour_list:
                    #if pixel is white/polymer
                    x1, y1, z1 = a[0], a[1], a[2]
                    # print('list = ',x1,y1,z1)
                    if np_array[x1,y1,z1] == 1: #if pixel is white
                        #get coordinate id
                        id_list.append(id_array[x1][y1][z1]) #np.array([x1, y1, z1])
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
                        if prop_dict[iid][4] == 1 and prop_dict[iid][0] == 0:
                            nc.append(iid)
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
                        x_x = int(center_coord_x - xx[0])
                        x_y = int(center_coord_y - xx[1])
                        x_z = int(center_coord_z - xx[2])
                        # x_list = [x_x, x_y, x_z]
                        key_matrix[1-x_x, 1-x_y, 1-x_z] = i
                        if bias is False:
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
                    path_df[j][t] = next_id
                    main_df.loc[path_df[j[t]], 'is_pathed?'] = 1
                    main_df.at[int(j[t]), 'is_pathed?'] = 1
                    print(path_df)
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
    np.save(output_dir+'path_array', np.asarray_chkfinite((path_array, prop_array, prop_dict, coords_dict, np_array, id_array)), allow_pickle=True)
    print('paths saved...')
    # np.save(output_dir+'prop_array', prop_array)
    return path_array, prop_array, prop_dict, coords_dict, np_array, id_array



if __name__ == "__main__":
    in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"
    out_dir = "/data/deg_test_output/"
    dir = pyd.Directory(in_dir, out_dir)
    input_dir = dir.InputDIR()
    output_dir = dir.OutputDIR()
    new_input_dir = pyd.Directory(out_dir).InputDIR()
    im = Voxelize_DF(input_dir)
    main_df = im.main_df()
    np_array = im.to_numpy()[0]
    id_array = im.vox_id()
    res = im.to_numpy()[1]
    im_class = InitPixelClassifier(main_df, np_array).init_classify(0.097, 0.67, 3)
    main_df = im_class[0]
    # print(im_class[0])
    im_path = PathArray(main_df, np_array).initPathArray()
    # print(im_path)
    path = iter_path(2, main_df, np_array, im_path, id_array, res, bias=True)
