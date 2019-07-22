from PIL import Image
import PIL.ImageOps as imgops
import scipy as sp
import numpy as np
import skimage as si
import pydirectory as pyd
import matplotlib.pyplot as plt
from porespy.tools import ps_disk, get_border
from scipy.signal import fftconvolve
import os

class Data():
    def __init__(self):
        self.data = {}

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

class Analyse():
    def __init__(self, data, sizes, mode):
        self.data = data
        self.sizes = sizes
        self.mode = mode
    def LocalThickness2(self, im):
        dt = sp.ndimage.distance_transform_edt(im)
        
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

    def Analyse(self):
        self.data["local_thickness"] = {}
        for i in self.data["filter"]:
            self.data["local_thickness"][i] = self.LocalThickness(self.data["filter"][i])
        
        return self.data

class Plot():
    # def __init__(self, data, layer):
    #     self.data = data
    #     # self.dims = dims
    #     # self.plotdims = plotdims
    #     # self.figsize = figsize
    #     self.layer = layer

    def __init__(self, data, layer, raw_data = True, filtered = True, local_thickness = True):
        self.data = data
        self.layer = layer
        self.raw_data = raw_data
        self.filtered = filtered
        self.local_thickness = local_thickness
    
    def PlotImage(self):
        # self.h, self.w = self.dims
        # self.nrows, self.ncols = self.plotdims
        # self.figsize = self.figsize
        truelist = list(filter(lambda x: x==True, [self.raw_data, self.filtered, self.local_thickness]))
        tlist = ["raw_data", "filtered", "local_thickness"]
        ncols = len(truelist)
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=[6,8])
        img1 = self.data["raw_data"][self.layer]
        img2 = self.data["filter"][self.layer]
        img3 = self.data["local_thickness"][self.layer]
        imgdata = []
        titlelist = []
        for a, b, c in zip([self.raw_data, self.filtered, self.local_thickness], [img1, img2, img3], tlist):
            if a == True:
                imgdata.append(b)
                titlelist.append(c)
        for i,j,k in zip(ax.flat,imgdata, titlelist):
            i.imshow(j, cmap='binary')
            i.set_title(k)
        plt.tight_layout(True)   
        # plt.figure()
        
        # plt.subplot(self.data["raw_data"][self.layer], cmap="gray")
        # plt.subplot(self.data["filter"][self.layer], cmap="gray")
        # plt.imshow(self.data["filter"][self.layer], cmap='binary')
        plt.show()

    def PlotImages(self):
        truelist = list(filter(lambda x: x==True, [self.raw_data, self.filtered, self.local_thickness]))
        ncols = len(truelist)
        # a = [False, True, False]
        # a= list(filter(lambda x: x == True, a))
        tlist = ["raw_data", "filtered", "local_thickness"]
        fig, ax = plt.subplots(nrows = 1, ncols = ncols, figsize = [6,8])
        img1 = self.data["raw_data"][self.layer]
        img2 = self.data["filter"][self.layer]
        img3 = self.data["local_thickness"][self.layer]
        imgdata = []
        titlelist = []
        for i, j, k in zip([self.raw_data, self.filtered, self.local_thickness], [img1, img2, img3], tlist):
            if i == True:
                imgdata.append(j)
                titlelist.append(k)
        for i, j, k in zip(ax.flat, imgdata, titlelist):
            i.imshow(j, cmap='binary')
            i.set_title(k)
        
        plt.tight_layout(True)
        plt.show()

class Plots():
    def __init__(self, data, img, raw_data = True, filtered = True, local_thickness = True):
        self.data = data
        self.img = img
        self.raw_data = raw_data
        self.filtered = filtered
        self.local_thickness = local_thickness

    def PlotImages(self):
        truelist = list(filter(lambda x: x==True, [self.raw_data, self.filtered, self.local_thickness]))
        ncols = len(truelist)
        # a = [False, True, False]
        # a= list(filter(lambda x: x == True, a))
        tlist = ["raw_data", "filtered", "local_thickness"]
        ax = plt.subplots(nrows = 1, ncols = ncols, figsize = [6,8])
        img1 = self.data["raw_data"][self.img]
        img2 = self.data["filter"][self.img]
        img3 = self.data["local_thickness"][self.img]
        imgdata = []
        titlelist = []
        for i, j, k in zip([self.raw_data, self.filtered, self.local_thickness], [img1, img2, img3], tlist):
            if i == True:
                imgdata.append(j)
                titlelist.append(k)
        for i, j, k in zip(ax.flat, imgdata, titlelist):
            i.imshow(j, cmap='binary')
            i.set_title(k)
        
        plt.tight_layout(True)
        plt.show()
        



if __name__ == "__main__":
    data = Data().data
    im = ImageImporter(data, '/data/downsample-2048-man-thres/', importall=False, layer="0-lx").Import()
    print(data)
    imf = Filters(data, 15, "median").ApplyFilter()
    print(data)
    lt = Analyse(data, sizes = 25, mode = "dt").Analyse()
    print(data)
    # Plot(imf, "0-lx").PlotImage()
    # lt = Analyse(imf, sizes = 25, mode = "dt").LocalThickness(imf["filter"]["0-lx"])
    # Plots(data = lt, img = "0-lx.tif").PlotImages()
    # Plot(data=lt, layer='0-lx.tif').PlotImages()
    # Plot(lt, layer = "0-lx").PlotImage()

