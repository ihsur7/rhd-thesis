import os
import math
import pydirectory as pyd
import numpy as np
import pandas as pd
import PIL.Image as Image
import random
import scipy.ndimage as ndimage
import pyvista
import PVGeo as pvgeo
from tqdm import tqdm
import gmpy2

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

def iter_path(main_df, np_array, path_df, id_array, res, bias = True, max_steps=None, save=False):
    #initialise 3D array shape of np_array with zeros that dynamically changes values
    #the values are probability numbers 
    #for each property array row, if voxel is active, run randomwalk
    prob_crys = 0.5
    prob_amorph = 1.5
    #POTENTIALLY ADD PATH SPLIT, use two random number generators between 0 and 1
    #first one tells which path splits
    #second one tells where it splits
    prob_self = 0.0 #0.001
    pm_p = 0.5 #0.1
    pm_a = 3 #2
    bug_check = False
    t = 1
    while np.all(main_df['is_pathed?']) == False:
        path_df = path_df.append(pd.DataFrame(np.zeros((1, path_df.shape[1]), dtype=np.int64)))
        path_df = path_df.reset_index(drop=True)
        for j in path_df:
            if bug_check and np.all(main_df['is_pathed?']):
                print('all elements pathed')
                break
            x = main_df.at[path_df[j][t-1], 'x']
            y = main_df.at[path_df[j][t-1], 'y']
            z = main_df.at[path_df[j][t-1], 'z']
            #gets coordinates of neighbouring voxels
            neighbour_list = neighbours(x, y, z, res = res)
            id_list = [path_df[j][t-1]]
            for a in neighbour_list:
                #if pixel is white/polymer
                x1, y1, z1 = a[0], a[1], a[2]
                if np_array[x1,y1,z1] == 1: #if pixel is white
                    #get coordinate id
                    id_list.append(id_array[x1][y1][z1])
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
                    if main_df.at[iid, 'is_crys'] == 1 and main_df.at[iid, 'is_pathed?'] == 0:
                        nc.append(iid)
                    elif main_df.at[iid, 'is_crys'] == 1 and main_df.at[iid, 'is_pathed?'] == 1:
                        npc.append(iid)
                    elif main_df.at[iid, 'is_crys'] == 0 and main_df.at[iid, 'is_pathed?'] == 0:
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
                pm_a *= pm_c
                ppa = pm_a*pm_p
                ppc = pm_c*pm_p
                center_id = id_list[0]
                center_coord_x = main_df.at[center_id, 'x']
                center_coord_y = main_df.at[center_id, 'y']
                center_coord_z = main_df.at[center_id, 'z']
                for i in id_list[1:]:
                    xx = main_df.at[i, 'x']
                    xy = main_df.at[i, 'y']
                    xz = main_df.at[i, 'z']
                    x_x = int(center_coord_x - xx)
                    x_y = int(center_coord_y - xy)
                    x_z = int(center_coord_z - xz)
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
                main_df.at[path_df[j][t], 'is_pathed?'] = 1
            else:
                single_px_counter += 1
        t += 1
        # print(path_df)
        if max_steps is not None and t == max_steps+1:
            break
    print('All voxels pathed.')
    print('Steps required: ', t)
    print('#single pixels: ', single_px_counter)
    if save:
        main_df.to_csv(output_dir+'main_df.csv')
        print("main_df saved...")
        path_df.to_csv(output_dir+'path_df.csv')
        print("path_df saved...")
        np.save(output_dir+'path_array', np.asarray_chkfinite((np_array, id_array), dtype=object), allow_pickle=True)
        print('arrays saved...')
    return main_df, path_df, np_array, id_array

def file_checker(input_dir, output_dir):
    im = Voxelize_DF(input_dir)
    np_array = im.to_numpy()[0]
    im_out = np.load(output_dir+"path_array.npy", allow_pickle=True)
    return 0 if np_array.all() == im_out[0].all() else 1

def Fick(diff, t, c0 = None, x=1):
    """
    returns C/C0
    C0 = water_conc

    """
    if c0 is not None:
        return c0*math.erfc(x/(math.sqrt(4*diff*t)))

    # print(conc_ratio)
    return math.erfc(x/(math.sqrt(4*diff*t)))

def MwLossData(temp, main_df, path_df, time_array, gradtype='lin', time_array_units = 'weeks'):
    # loss_rate= [0.001776, 0.002112, 0.002527] #ln, in weeks too high, may need to be in seconds
    loss_rate = [4.900e-008, 4.737e-008, 4.262e-008] #ln, in seconds
    # loss_rate = [0.0007610, 0.0008171, 0.001194]
    # loss_rate = [0.0002538, 0.0003017, 0.0003611] #ln, in days   
    print("gradtype = ", gradtype) 
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
    data_arr = np.zeros(shape=(len(main_df.index), len(time_array)))
    data_df = pd.DataFrame(data_arr, columns = ['t'+str(i) for i in list(range(len(time_array)))])
    conc_df = data_df
    conc_df_1 = conc_df
    for i in data_df.index:
        data_df['t0'][i] = main_df['mw'][i]
        conc_df['t0'][i] = main_df['water_conc'][i]
        # data_df.loc[[i, 't0']]= main_df.loc[[i, 'mw']]
        # conc_df.loc[[i, 't0']] = main_df.loc[[i, 'water_conc']]
    for tindex, t in enumerate(tqdm(time_array)):
        for p in path_df: #for column in path_df
            for index, q in enumerate(path_df[p]):
                if time_array_units == 'weeks':
                    tt = t*604800 #weeks in seconds
                elif time_array_units == 'days':
                    tt = t*(604800/7)
                elif time_array_units == 'hours':
                    tt = t*3600
                elif time_array_units == 'minutes':
                    tt = t*60
                elif time_array_units == 'seconds': 
                    tt = t
                else:
                    print('Unknown time units')
                    break
                if t == 0:
                    avg_conc = 0
                else:
                    avg_conc = (Fick(diff_coeff_mm["37"], tt, c0=water_conc, x=index*pixel_scale) + Fick(diff_coeff_mm["37"], tt, c0=water_conc, x=(index+1)*pixel_scale))/2
                total_conc = gmpy2.sqrt((avg_conc**2)+(main_df.loc[q, 'water_conc']**2))
                # print(total_conc)
                # total_conc = math.sqrt((avg_conc**2)+(path[2][q][3]**2))
                total_conc_ratio = total_conc/water_conc
                conc_df.loc[q, 't'+str(tindex)] = total_conc_ratio
                # conc_dict[q][tindex] = total_conc_ratio
                main_df.loc[q, 'water_conc'] = total_conc_ratio
                # path[2][q][3] = total_conc_ratio
                # path[1][0][index+1][4] = total_conc_ratio
                conc_df_1.loc[q, 't'+str(tindex)] += total_conc_ratio
                # conc_data_dict_1[q][tindex] += total_conc_ratio
                if gradtype == "lin":
                    multiplier = total_conc_ratio
                    # grad = loss_rate_calc(average_loss_rate, avg_conc_ratio)
                elif gradtype == "exp":
                    a = 0.05
                    multiplier = a*math.e**(math.log((1/a)+1)*total_conc_ratio)-1
                elif gradtype == 'log':
                    multiplier = (1/math.log10(2))*math.log10(total_conc_ratio+1)
                elif gradtype == 'quad':
                    a = -1
                    b = 1
                    c = 1
                    multiplier = a*(total_conc_ratio**2) + b*total_conc_ratio + c
                else:
                    multiplier = 1
                loss_rate = average_loss_rate**multiplier
                if main_df['is_crys'][q] == 1:
                # if path[2][q][4] == 1:
                    loss_rate *= loss_rate
                mwt = main_df['mw'][q]*math.e**(-1*(loss_rate)*tt)
                # mwt = path[2][q][5]*math.e**(-1*(loss_rate)*tt)
                data_df.loc[q, 't'+str(tindex)] = mwt
                # data_dict[q][tindex] = mwt
                # data_array[index][tindex+1] = mwt
    main_df.to_csv(output_dir+'main_df.csv')
    print("main_df saved...")
    path_df.to_csv(output_dir+'path_df.csv')
    print("path_df saved...")
    data_df.to_csv(output_dir+'data_df.csv')
    print("data_df saved...")
    conc_df.to_csv(output_dir+'conc_df.csv')
    print("conc_df saved...")
    # np.save(output_dir+'conc_ratio'+'_'+gradtype, conc_data_array, allow_pickle=True)
    return main_df, path_df, data_df

class Visualise2:
    def __init__(self, main_df, data_df, time_array):
        self.main_df = main_df
        self.data_df = data_df
        self.time_array = time_array
coords_df = main_df[['x', 'y', 'z']].copy()
    # print(data_df)
    for i in ['t'+str(i) for i in list(range(len(time_array)))]:
        coords_df = coords_df.join(data_df[i])
    # coords_df = coords_df.join(data_df)
    # print(coords_df)
    vtkpoints = pvgeo.points_to_poly_data(coords_df)
    # vtkpoints = pvgeo.points_to_poly_data(model)
    # print(vtkpoints)
    bounds = vtkpoints.bounds
    # print(bounds)
    margin = 100
    n = 300 #600
    ldim = bounds[-1] + margin*2
    grid = pyvista.UniformGrid((n,n,n))
    grid.origin = [bounds[0] - margin]*3
    spacing = ldim/(n-1)
    grid.spacing = [spacing]*3

    vox = grid.interpolate(vtkpoints,radius=spacing*2,progress_bar=True)
    mask = vox['t0']>0
    vox_valid = vox.extract_points(mask, adjacent_cells=False)
    # vox_valid.plot()
    plotter = pyvista.Plotter(notebook=False)
    plotter.add_mesh(vox_valid)
    plotter.add_slider_widget(change_t, [0,len(time_array)-1], title='Time')
    plotter.show()
        

def Visualise(main_df, data_df, time_array, scalar=None):
    # pv.set_plot_theme('dark')

    # pcloud = pv.PolyData(model)
    # # print(pcloud.n_points())
    # pcloud['radius'] = np.asarray([1]*coords.shape[0])

    # geom1 = pv.Cube()
    # # geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    # glyphed = pcloud.glyph(scale="radius", geom=geom1) # progress_bar=True)
    # pcloud.point_data['scalars'] = scalar
    # pcloud.set_active_scalars('scalars')
    # # print(glyphed.n_points())
    

    # p = pv.Plotter(notebook=False)
    # p.add_mesh(glyphed, show_edges=True, edge_color='black', scalars='scalars')
    # print(pcloud)
    # # print(p.n_points())
    # p.show()
    #https://github.com/pyvista/pyvista-support/issues/346
    # pyvista.rcParams['use_ipyvtk'] = True
    coords_df = main_df[['x', 'y', 'z']].copy()
    # print(data_df)
    for i in ['t'+str(i) for i in list(range(len(time_array)))]:
        coords_df = coords_df.join(data_df[i])
    # coords_df = coords_df.join(data_df)
    # print(coords_df)
    vtkpoints = pvgeo.points_to_poly_data(coords_df)
    # vtkpoints = pvgeo.points_to_poly_data(model)
    # print(vtkpoints)
    bounds = vtkpoints.bounds
    # print(bounds)
    margin = 100
    n = 300 #600
    ldim = bounds[-1] + margin*2
    grid = pyvista.UniformGrid((n,n,n))
    grid.origin = [bounds[0] - margin]*3
    spacing = ldim/(n-1)
    grid.spacing = [spacing]*3

    vox = grid.interpolate(vtkpoints,radius=spacing*2,progress_bar=True)
    mask = vox['t0']>0
    vox_valid = vox.extract_points(mask, adjacent_cells=False)
    # vox_valid.plot()
    plotter = pyvista.Plotter(notebook=False)
    plotter.add_mesh(vox_valid)
    plotter.add_slider_widget(change_t, [0,len(time_array)-1], title='Time')
    plotter.show()

if __name__ == "__main__":
    in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"
    out_dir = "/data/deg_test_output/new/"
    pdir = pyd.Directory(in_dir, out_dir)
    input_dir = pdir.InputDIR()
    output_dir = pdir.OutputDIR()
    new_input_dir = pyd.Directory(out_dir).InputDIR()
    im = Voxelize_DF(input_dir)
    main_df = im.main_df()
    np_array = im.to_numpy()[0]
    id_array = im.vox_id()
    res = im.to_numpy()[1]
    im_class = InitPixelClassifier(main_df, np_array).init_classify(0.097, 0.67, 3)
    main_df = im_class[0]
    # print(im_class[0])
    if file_checker(input_dir, output_dir) == 1:
        print("file not found. paths will be generated")
        im_path = PathArray(main_df, np_array).initPathArray()
        # print(im_path)
        path = iter_path(main_df, np_array, im_path, id_array, res, bias=True, max_steps = None, save=True)
        path_df = path[1]
    else:
        print("file exists. no paths will be generated")
        path_df = pd.read_csv(output_dir+'path_df.csv')
        path = np.load(output_dir+"path_array.npy", allow_pickle=True)
        print('model resolution = ', path[0].shape)
    print('# polymer voxels = ', main_df.shape[0])
    #25 = 211.88 seconds on macbook, ~50k paths (flow vectors)
    #50 = 
    time_array = np.arange(start=0, stop=2, step=1)
    print('# timepoints: ', time_array.shape[0], '\ntimepoints: ', time_array)
    diff_coeff = {"25": 51.7e-12, "37": 67.6e-12, "50": 165e-12} #x10^(-12) m^2/s
    diff_coeff_mm = {"25": 51.7e-5, "37": 67.6e-5, "50": 165e-5} #mm^2/s
    # print(diff_coeff_mm)
    pixel_scale = 0.25/35 #mm/px * 1px
    temp = '37'
    # iter_fick(300, temp, pixel_scale, coords, mat_props, nparr, idarr, flowpath)
    #Calculate Flowpath
    gradtypelist = ['lin', 'exp', 'log', 'quad']
    tmu = 'weeks'
    gtype = 'lin' #default = lin
    print('Calculating Mw data...')
    # mw_data = np.load(output_dir+'data_array'+'_'+gtype+'.npy', allow_pickle=True)
    # mw_data = MwLossData("37", main_df, path_df, time_array, gradtype=gtype, time_array_units=tmu)
    data_df = pd.read_csv(output_dir+'data_df.csv')
    avg_mw = []
    # print(mw_data)
    for i in ['t'+str(i) for i in list(range(len(time_array)))]:
        # print(i)
        # print(data_df[i].mean())
        avg_mw.append(data_df[i].mean())

    print(avg_mw)
    Visualise(main_df, data_df, time_array)
    # avg_mw = [i for i in mw_data[2][i].mean()]
