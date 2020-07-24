from PIL import Image
import PIL.ImageOps as imgops
import scipy as sp
import scipy.ndimage
import numpy as np
import skimage as si
import pydirectory as pyd
import porespy as ps
import matplotlib.pyplot as plt
from porespy.tools import ps_disk, get_border
from scipy.signal import fftconvolve
import os


class Data:
    def __init__(self):
        self.data = {}
        self.preferences = {}


class ImageImporter:
    def __init__(self, data, inputdir, importall=False, layer=None):
        self.inputdir = inputdir
        self.importall = importall
        self.layer = layer
        self.data = data

    def import_image(self):
        imdir = pyd.Directory(self.inputdir).InputDIR()
        print(imdir)
        self.data["raw_data"] = {}
        print(self.data)
        if self.importall:
            listdir = os.listdir(imdir)
            for i in listdir:
                self.data["raw_data"][i[:-4]] = np.array(Image.open(pyd.Directory(self.inputdir + i).InputDIR()))
            # print(self.data)
        elif self.importall == False and self.layer is None:
            raise Exception("Layer not provided!")
        elif self.importall == True and self.layer is not None:
            raise Exception("Layer should not be included!")
        else:
            self.data["raw_data"][self.layer] = np.array(
                Image.open(pyd.Directory(self.inputdir + self.layer + '.tif').InputDIR()))
        return self.data


class Filters:
    def __init__(self, data, min_size, ftype="median"):
        self.data = data
        self.min_size = min_size
        self.ftype = ftype

    def median_filter(self):
        self.data["filter"] = {}
        for i in self.data["raw_data"]:
            self.data["filter"][i] = sp.ndimage.median_filter(self.data["raw_data"][i], size=self.min_size)
        return self.data

    def maximum_filter(self):
        self.data["filter"] = {}
        for i in self.data["raw_data"]:
            self.data["filter"][i] = sp.ndimage.maximum_filter(self.data["raw_data"][i], size=self.min_size)
        return self.data

    def apply_filter(self):
        if self.ftype == "median":
            return self.median_filter()
        elif self.ftype == "maximum":
            return self.maximum_filter()
        else:
            raise Exception("Unknown filter " + self.ftype)


class AnalyseImage():
    def __init__(self, data, sizes):
        self.data = data
        self.sizes = sizes

    def get_edges(self, shape, thickness=1, return_indices=False):
        t = thickness
        border = sp.ones(shape, dtype=bool)
        border[t:-t, t:-t] = False
        if return_indices:
            border = sp.where(border)
        return border

    def get_disk(self, radius):
        rad = int(sp.ceil(radius))
        other = sp.ones((2 * rad + 1, 2 * rad + 1), dtype=bool)
        other[rad, rad] = False
        disk = sp.ndimage.distance_transform_edt(other) < radius
        return disk

    def local_thickness(self, im):
        dt = sp.ndimage.distance_transform_edt(im)
        if isinstance(self.sizes, int):
            self.sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=self.sizes)
        else:
            self.sizes = sp.unique(self.sizes)[-1::-1]
        imresults = sp.zeros(sp.shape(im))
        for r in self.sizes:
            imtemp = dt >= r
            if sp.any(imtemp):
                imtemp = sp.ndimage.distance_transform_edt(~imtemp) < r
                imresults[(imresults == 0) * imtemp] = r
        return imresults

    def analyse(self, measure="material"):
        self.data["local_thickness"] = {}
        self.data["inverted_image"] = {}
        if measure == "pores":
            for i in self.data["filter"]:
                self.data["inverted_image"][i] = np.array(imgops.invert(Image.fromarray(self.data["filter"][i])))
                self.data["local_thickness"][i] = self.local_thickness(self.data["inverted_image"][i])
        elif measure == "material":
            for i in self.data["filter"]:
                print('processing layer ... ' + self.data["layer"][i])
                self.data["local_thickness"][i] = self.local_thickness(self.data["filter"][i])
        else:
            raise Exception("Unknown measure: " + measure)
        return self.data


class Plot:
    def __init__(self, data, layer):
        self.data = data
        self.layer = layer

    def plot_images(self, raw_data=True, filtered=True, local_thickness=True):
        tlist = [raw_data, filtered, local_thickness]
        numtrue = len(list(filter(lambda x: x == True, tlist)))
        titlelist = ["raw_data", "filter", "local_thickness"]
        fig, ax = plt.subplots(nrows=1, ncols=numtrue, figsize=[8, 8])
        imgdata = []
        titlelist2 = []
        for i, j in zip(tlist, titlelist):
            if i:
                imgdata.append(self.data[j][self.layer])
                titlelist2.append(j)
        for a, b, c in zip(ax.flat, imgdata, titlelist2):
            a.imshow(b, cmap="binary")
            a.set_title(c)
        plt.tight_layout(True)
        plt.show()
        # print(titlelist2, imgdata)

    def plot_chart(self, ctype="line", dtype="cdf"):

        return


class Statistics():
    def __init__(self, data):
        self.data = data

    def get_porosity(self):
        self.data["porosity"] = {}
        for i in self.data["filter"]:
            self.data["porosity"][i] = {}
            im = sp.array(self.data["filter"][i], dtype=bool)
            pore = sp.sum(im == 0)
            surf = sp.sum(im == 1)
            porosity = round(pore / (surf + pore), 3)
            surface = round(1.0 - porosity, 3)

            self.data["porosity"][i]["surface"] = surface
            self.data["porosity"][i]["pores"] = porosity
        print(self.data["porosity"])
        return self.data

    def pore_distribution(self, bins=10, log=True, voxel_size=1):
        self.data["psd"] = {}
        for i in self.data["local_thickness"]:
            # voxel_size = self.data["raw_data"][i].shape
            # scaling = 774/5278
            # voxel_size=1/ (self.data["raw_data"][i].shape[0] * scaling)
            # scaling = self.data["raw_data"][i].shape[0] / 5278  # 2048/5278
            # pixel_scale = 1 / 774  # mm/px
            # voxel_size = pixel_scale * scaling
            im = self.data["local_thickness"][i]
            dat = ps.metrics.pore_size_distribution(im=im, bins=bins, log=log, voxel_size=voxel_size)
            self.data["psd"][i] = dat
        return self.data

def saveCSV(output):
    return

if __name__ == "__main__":
    data = Data().data
    prefs = Data().preferences
    prefs["input"] = '/data/downsample-2048-man-thres/nx-4/'
    prefs["log"] = False
    prefs["layer"] = ['nx-12'] #, '4-lx', '8-lx', '12-lx', '16-lx', '20-lx']
    prefs["sizes"] = 50
    prefs["bins"] = int(prefs["sizes"] / 2)
    print("Preferences: {}".format(prefs))
    data = ImageImporter(data, prefs["input"], layer=prefs["layer"][0]).import_image()
    print(data)
    imf = Filters(data, 1, "median").apply_filter()
    # print(data)
    lt = AnalyseImage(data, sizes=prefs["sizes"]).analyse()
    # print(data)
    # Plot(data, prefs["layer"]).plot_images()
    stats = Statistics(data)
    # stats.GetPorosity()
    stats.pore_distribution(bins=prefs["bins"], log=prefs["log"])
    rlist = [data['psd'][prefs['layer'][0]].R]
    pdflist = [data['psd'][prefs['layer'][0]].pdf]
    sumlist = np.sum(np.asarray([i * j for i, j in zip(rlist, pdflist)]))
    print('Avg. pore size for {}: {}'.format(prefs["layer"][0], sumlist))

    # print(data["psd"])
    plt.plot(data["psd"]["0-lx"].R, data["psd"]["0-lx"].cdf)
    plt.show()
    # print(data["psd"])
