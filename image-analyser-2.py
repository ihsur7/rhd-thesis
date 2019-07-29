from PIL import Image
import PIL.ImageOps as imgops
import scipy as sp
import numpy as np
import skimage as si
import pydirectory as pyd
import porespy as ps
import matplotlib.pyplot as plt
from porespy.tools import ps_disk, get_border
from scipy.signal import fftconvolve
import os

class Data():
    def __init__(self):
        self.data = {}
        self.preferences = {}

class ImageImporter():
    def __init__(self, data, inputdir, importall = False, layer = None):
        self.inputdir = inputdir
        self.importall = importall
        self.layer = layer
        self.data = data

    def Import(self):
        imdir = pyd.Directory(self.inputdir).InputDIR()
        self.data["raw_data"] = {}
        if self.importall == True:
            listdir = os.listdir(imdir)
            for i in listdir:
                self.data["raw_data"][i[:-4]] = np.array(Image.open(pyd.Directory(self.inputdir+i).InputDIR()))
            # print(self.data)
        elif self.importall == False and self.layer == None:
            raise Exception("Layer not provided!")
        elif self.importall == True and self.layer is not None:
            raise Exception("Layer should not be included!")
        else:
            self.data["raw_data"][self.layer] = np.array(Image.open(pyd.Directory(self.inputdir+self.layer+'.tif').InputDIR()))
        return self.data

class Filters():
    def __init__(self, data, min_size, ftype = "median"):
        self.data = data
        self.min_size = min_size
        self.ftype = ftype
    
    def MedianFilter(self):
        self.data["filter"] = {}
        for i in self.data["raw_data"]:
            self.data["filter"][i] = sp.ndimage.median_filter(self.data["raw_data"][i], size = self.min_size)
        # sp.ndimage.median_filter((self.data["filter"][i] for i in self.data["raw_data"]), size = self.min_size)
        # print(self.data)
        return self.data
    
    def MaximumFilter(self):
        self.data["filter"] = {}
        for i in self.data["raw_data"]:
            self.data["filter"][i] = sp.ndimage.maximum_filter(self.data["raw_data"][i], size = self.min_size)
        return self.data
    
    def ApplyFilter(self):
        if self.ftype == "median":
            return self.MedianFilter()
        elif self.ftype == "maximum":
            return self.MaximumFilter()
        else:
            raise Exception("Unknown filter " + self.ftype)

class AnalyseImage():
    def __init__(self, data, sizes, mode):
        self.data = data
        self.sizes = sizes
        self.mode = mode
    
    def GetEdges(self, shape, thickness=1, return_indices=False):
        t = thickness
        border = sp.ones(shape, dtype=bool)
        border[t:-t, t:-t] = False
        if return_indices:
            border = sp.where(border)
        return border
    
    def GetDisk(self, radius):
        rad = int(sp.ceil(radius))
        other = sp.ones((2 * rad + 1, 2 * rad + 1), dtype=bool)
        other[rad, rad] = False
        disk = sp.ndimage.distance_transform_edt(other) < radius
        return disk

    def LocalThickness2(self, im):
        dt = sp.ndimage.distance_transform_edt(im)
        inlets = self.GetEdges(im.shape)
        if isinstance(self.sizes, int):
            self.sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=self.sizes)
        else:
            self.sizes = sp.unique(self.sizes)[-1::-1]

        strel = self.GetDisk

        inlets = sp.where(inlets)
        imresults = sp.zeros(sp.shape(im))
        if self.mode == "dt":
            inlets = sp.where(inlets)
            imresults = sp.zeros(sp.shape(im))
            for r in self.sizes:
                imtemp = dt >= r
                if sp.any(imtemp):
                    imtemp = sp.ndimage.distance_transform_edt(~imtemp) < r
                    imresults[(imresults == 0)*imtemp] = r
        elif self.mode == "hybrid":
            inlets = sp.where(inlets)
            imresults = sp.zeros(sp.shape(im))
            for r in self.sizes:
                imtemp = dt >= r
                if sp.any(imtemp):
                    imtemp = fftconvolve(imtemp, strel(r), mode = 'same') > 0.0001
                    imresults[(imresults == 0)*imtemp] = r
        else:
            raise Exception('Unknown Mode + ' + self.mode)

        return imresults

    def LocalThickness(self, im, inlets = None):
        #based on Porespy
        dt = sp.ndimage.distance_transform_edt(im > 0)

        if inlets is None:
            inlets = get_border(im.shape, mode = 'faces')

        if isinstance(self.sizes, int):
            self.sizes = sp.logspace(start = sp.log10(sp.amax(dt)), stop = 0, num = self.sizes)
        else:
            self.sizes = sp.unique(self.sizes)[-1::-1]

        strel = ps_disk

        if self.mode == "dt":
            inlets = sp.where(inlets)
            imresults = sp.zeros(sp.shape(im))
            for r in self.sizes:
                imtemp = dt >= r
                if sp.any(imtemp):
                    imtemp = sp.ndimage.distance_transform_edt(~imtemp) < r
                    imresults[(imresults == 0)*imtemp] = r
        elif self.mode == "hybrid":
            inlets = sp.where(inlets)
            imresults = sp.zeros(sp.shape(im))
            for r in self.sizes:
                imtemp = dt >= r
                if sp.any(imtemp):
                    imtemp = fftconvolve(imtemp, strel(r), mode = 'same') > 0.0001
                    imresults[(imresults == 0)*imtemp] = r
        else:
            raise Exception('Unknown Mode + ' + self.mode)

        return imresults

    def Analyse(self, measure="pores"):
        self.data["local_thickness"] = {}
        self.data["inverted_image"] = {}
        if measure=="pores":
            for i in self.data["filter"]:
                self.data["inverted_image"][i] = np.array(imgops.invert(Image.fromarray(self.data["filter"][i])))
                self.data["local_thickness"][i] = self.LocalThickness2(self.data["inverted_image"][i])
        elif measure=="material":
            for i in self.data["filter"]:
                self.data["local_thickness"][i] = self.LocalThickness2(self.data["filter"][i])
        else:
            raise Exception("Unkown measure: "+measure)
        return self.data

class Plot():
    def __init__(self, data, layer):
        self.data = data
        self.layer = layer

    def Plot(self, raw_data=True, filtered=True, local_thickness=True):
        tlist = [raw_data, filtered, local_thickness]
        numtrue = len(list(filter(lambda x: x == True, tlist)))
        titlelist = ["raw_data", "filter", "local_thickness"]
        fig, ax = plt.subplots(nrows=1, ncols=numtrue, figsize=[8,8])
        imgdata = []
        titlelist2 = []
        for i, j in zip(tlist, titlelist):
            if i == True:
                imgdata.append(self.data[j][self.layer])
                titlelist2.append(j)
        for a, b, c in zip(ax.flat, imgdata, titlelist2):
            a.imshow(b, cmap="binary")
            a.set_title(c)
        plt.tight_layout(True)
        plt.show()
        # print(titlelist2, imgdata)
    def PlotChart(self, ctype="line", dtype="cdf"):
                
        return

class Statistics():
    def __init__(self, data):
        self.data = data
    
    def GetPorosity(self):
        self.data["porosity"] = {}
        for i in self.data["filter"]:
            self.data["porosity"][i] = {}
            im = sp.array(self.data["filter"][i], dtype=bool)
            pore = sp.sum(im == 0)
            surf = sp.sum(im == 1)
            porosity = round(pore/(surf + pore), 3)
            surface = round(1.0 - porosity, 3)
            
            self.data["porosity"][i]["surface"] = surface
            self.data["porosity"][i]["pores"] = porosity
        print(self.data["porosity"])
        return self.data
    
    def PoreDistribution(self, bins=10, log=True, voxel_size=1):
        self.data["psd"] = {}
        for i in self.data["local_thickness"]:
            # voxel_size = self.data["raw_data"][i].shape
            scaling = 774/5278
            voxel_size=1/ (self.data["raw_data"][i].shape[0] * scaling)
            im = self.data["local_thickness"][i]
            dat = ps.metrics.pore_size_distribution(im=im, bins=bins, log=log, voxel_size=voxel_size)
            self.data["psd"][i] = dat
        return self.data




if __name__ == "__main__":
    data = Data().data
    prefs = Data().preferences
    prefs["input"] = '/data/downsample-2048-man-thres/'
    prefs["log"] = False
    prefs["layer"] = '0-lx'
    prefs["sizes"] = 150
    prefs["mode"] = 'hybrid'
    prefs["bins"] = int(prefs["sizes"]/2)
    print(prefs)
    im = ImageImporter(data, prefs["input"], importall=False, layer=prefs["layer"]).Import()
    # print(data)
    imf = Filters(data, 15, "median").ApplyFilter()
    # print(data)
    lt = AnalyseImage(data, sizes = prefs["sizes"], mode = prefs["mode"]).Analyse()
    # print(data)
    stats = Statistics(data)
    # stats.GetPorosity()
    stats.PoreDistribution(bins=prefs["bins"], log=prefs["log"])
    # print(data["psd"])
    plt.plot(data["psd"]["0-lx"].R, data["psd"]["0-lx"].cdf)
    plt.show()
    # print(data["psd"])