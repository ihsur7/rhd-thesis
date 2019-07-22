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

class AnalyseImage():
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

    def Analyse(self, measure="pores"):
        self.data["local_thickness"] = {}
        self.data["inverted_image"] = {}
        if measure=="pores":
            for i in self.data["filter"]:
                self.data["inverted_image"][i] = np.array(imgops.invert(Image.fromarray(self.data["filter"][i])))
                self.data["local_thickness"][i] = self.LocalThickness(self.data["inverted_image"][i])
        elif measure=="material":
            for i in self.data["filter"]:
                self.data["local_thickness"][i] = self.LocalThickness(self.data["filter"][i])
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
            a.imshow(b)
            a.set_title(c)
        plt.tight_layout(True)
        plt.show()
        # print(titlelist2, imgdata)

class Statistics():
    def __init__(self, data):
        self.data = data
    
    


if __name__ == "__main__":
    data = Data().data
    im = ImageImporter(data, '/data/downsample-2048-man-thres/', importall=False, layer="0-lx").Import()
    # print(data)
    imf = Filters(data, 15, "median").ApplyFilter()
    # print(data)
    lt = AnalyseImage(data, sizes = 25, mode = "dt").Analyse()
    # print(data)
    Plot(data, layer = '0-lx').Plot()
    