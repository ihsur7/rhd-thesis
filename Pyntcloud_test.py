#Pyntcloud test.py

import pydirectory as pyd
import numpy as np
import PIL.Image as Image
import os
import pyvista as pv

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

in_dir = "/data/sample1/50/"

input_dir = pyd.Directory(in_dir).InputDIR()

a = Voxelize(input_dir)
coords_id = a.coord_array()
coords = np.delete(coords_id,0,1)
# print(coords.shape)

pv.set_plot_theme('dark')

pcloud = pv.PolyData(coords)
pcloud['radius'] = np.asarray([1]*coords.shape[0])
geom1 = pv.Cube()
# geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
glyphed = pcloud.glyph(scale="radius", geom=geom1) # progress_bar=True)

p = pv.Plotter(notebook=False)
p.add_mesh(glyphed, color='white', show_edges=True, edge_color='black')
p.show()

# print(pcloud)
# pcloud.plot(eye_dome_lighting=True, render_points_as_spheres=True)