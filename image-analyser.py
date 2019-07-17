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


class ImageImporter():
    def __init__(self, inputdir):
        self.inputdir = inputdir
        self.data = {}

    def Import(self):
        imdir = pyd.Directory(self.inputdir).InputDIR()
        listdir = os.listdir(imdir)
        self.data["raw_data"] = {}
        for i in listdir:
            self.data["raw_data"][i] = np.array(Image.open(pyd.Directory(self.inputdir+i).InputDIR()))
        # print(self.data)
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

class Plot():
    def __init__(self, data, layer):
        self.data = data
        # self.dims = dims
        # self.plotdims = plotdims
        # self.figsize = figsize
        self.layer = layer
    
    def PlotImage(self):
        # self.h, self.w = self.dims
        # self.nrows, self.ncols = self.plotdims
        # self.figsize = self.figsize
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[6,8])
        img1 = self.data["raw_data"][self.layer+'.tif']
        img2 = self.data["filter"][self.layer+'.tif']
        imgdata = [img1, img2]
        titlelist = ["raw_data", "filter"]
        for i,j,k in zip(ax.flat,imgdata, titlelist):
            i.imshow(j, cmap='binary')
            i.set_title(k)
        plt.tight_layout(True)   
        # plt.figure()
        
        # plt.subplot(self.data["raw_data"][self.layer], cmap="gray")
        # plt.subplot(self.data["filter"][self.layer], cmap="gray")
        # plt.imshow(self.data["filter"][self.layer], cmap='binary')
        plt.show()

class Analyse():
    def __init__(self, data, sizes, mode):
        self.data = data
        self.sizes = sizes
        self.mode = mode

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

    def test(self):
        self.data["local_thickness"] = {}
        for i in self.data["filter"]:
            self.data["local_thickness"][i] = self.LocalThickness(self.data["filter"][i])
        
        return self.data


if __name__ == "__main__":
    im = ImageImporter('/data/downsample-2048-man-thres/').Import()
    imf = Filters(im, 15, "median").ApplyFilter()
    # Plot(imf, "0-lx").PlotImage()
    lt = Analyse(imf, sizes = 25, mode = "hybrid").LocalThickness(imf["filter"]["0-lx.tif"])

