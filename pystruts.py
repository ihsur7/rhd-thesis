from PIL import Image
import PIL.ImageOps as imgops
import scipy as sp
import numpy as np
import scipy.ndimage as ndimage
import skimage
import skimage.morphology as morph
import pydirectory as pyd
import matplotlib.pyplot as plt
import os
import csv 

def convert_bool(im):
    return np.array(im, dtype=bool)

def _parse_data(data):
    im = list(data)[0]
    del_list = ['raw_data', 'filter', 'lt']
    new_data = {}
    for i in del_list:
        del data[im][i]
    psd_data = list(data[im]['psd'])
    for i in psd_data:
        data[im][i] = data[im]['psd'][i]
    for i in ['white', 'black']:
        data[im][i] = data[im]['porosity'][i]
        # new_data[i] = data[im][i]
    del data[im]['porosity']
    data[im].pop('psd')
    newlist = list(data[im])[:-2]
    for i in newlist:
        new_data[i] = np.ndarray.tolist(data[im][i])
    return new_data

def save_csv(data):
    # bin_centers = R
    im = list(data)[0]
    data = _parse_data(data)
    csv_cols = list(data)
    filename = im+'.csv'
    # print(data.values())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

class Data(object):
    def __init__(self):
        self.data = {}
        self.preferences = {}
    
class ImageImporter(object):
    def __init__(self, data, inputdir):
        self.data = data
        self.inputdir = inputdir
    
    def import_image(self, image):
        self.data[image] = {}
        self.data[image]["raw_data"] = np.array(Image.open(
            pyd.Directory(self.inputdir + image + '.tif').InputDIR()
        ))
        return self.data
        
class Filters(object):
    def __init__(self, data, min_size):
        self.data = data
        self.min_size = min_size

    def median_filter(self):
        im = list(self.data)[0]
        # for i in self.data:
        self.data[im]["filter"] = ndimage.median_filter(
            self.data[im]["raw_data"], size = self.min_size
        )
        return self.data

class LocalThickness(object):
    def __init__(self, data):
        self.data = data
        self.im = list(self.data)[0]

    def distance_transform(self, im):
        return ndimage.distance_transform_edt(im)
        
    def skeleton(self):
        self.data[self.im]["skeleton"] = morph.skeletonize(convert_bool(self.data[self.im]["filter"]))
        self.data[self.im]["watershed"] = morph.watershed(self.data[self.im]["filter"])
        return self.data

    def ridge(self):
        skeletonize = self.skeleton()
        sk = self.data[self.im]["skeleton"]
        loc = np.transpose(np.nonzero(sk))
        imresults = np.zeros(shape=self.data[self.im]["raw_data"].shape)
        dt = ndimage.distance_transform_edt(self.data[self.im]["filter"])
        # print(loc)
        for i,j in loc:
            imresults[i,j] = dt[i,j]
        self.data[self.im]["ridge"] = imresults
        # print(self.data[self.im]["filter"])

    def local_thickness(self, sizes, invert=False):
        if invert:
            im_invert = np.array(imgops.invert(Image.fromarray(
                self.data[self.im]["filter"])))
            dt = ndimage.distance_transform_edt(im_invert)
        else:
            dt = ndimage.distance_transform_edt(
                self.data[self.im]["filter"])
        if isinstance(sizes, int):
            sizes = np.logspace(start = np.log10(np.amax(dt)), stop = 0, num = sizes)
        else:
            sizes = np.unique(sizes)[-1::-1]
        imresults = np.zeros(shape=self.data[self.im]["raw_data"].shape)
        for r in sizes:
            imtemp = dt >= r
            if np.any(imtemp):
                imtemp = ndimage.distance_transform_edt(~imtemp) < r
                imresults[(imresults == 0)*imtemp] = r
        self.data[self.im]["lt"] = imresults
        return self.data
        
class Measure(object):
    def __init__(self, data):
        self.data = data
        # print(self.data)
        self.im = list(self.data)[0]

    def _parse_histogram(self, h, voxel_size=1):
        dx = h[1]
        P = h[0]
        temp = P*(dx[1:] - dx[:-1])
        C = np.cumsum(temp[-1::-1])[-1::-1]
        S = P*(dx[1:] - dx[:-1])
        bin_edges = dx * voxel_size
        bin_widths = (dx[1:] - dx[:-1]) * voxel_size
        bin_centers = ((dx[1:] + dx[:-1])/2) * voxel_size
        psd = {'pdf': P, 'cdf': C, 'relfreq': S,\
               'bin_centers': bin_centers, 'bin_edges': bin_edges,\
               'bin_widths': bin_widths}
        return psd
    
    def porosity(self):
        image = self.data[self.im]["filter"]
        wh = list(np.array(image, dtype=bool).flatten()).count(0)
        w, h = image.shape
        white = wh/(w*h)
        black = 1 - white
        self.data[self.im]["porosity"] = {'white': white, 'black': black}
        return self.data

    def psd(self, voxel_size, bins=10, log=False):
        im = self.data[self.im]["lt"].flatten()
        vals = im[im > 0] * voxel_size
        if log:
            vals = np.log10(vals)
        h = self._parse_histogram(np.histogram(vals, bins=bins, density=True))
        psd = {'pore_size_distribution': {log*'log' + 'R': h['bin_centers'],\
               'pdf': h['pdf'], 'cdf': h['cdf'], 'satn': h['relfreq'],\
               'bin_centers': h['bin_centers'], 'bin_edges': h['bin_edges'],\
               'bin_widths': h['bin_widths']}}
        self.data[self.im]['psd'] = psd['pore_size_distribution']
        return self.data
    
    def measure_all(self, voxel_size, bins, log):
        self.porosity()
        self.psd(voxel_size, bins, log)
        
    