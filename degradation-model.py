import pydirectory as pyd
import open3d as otd
import numpy as np
# import stl
import numpy
import random
import scipy as sp
import scipy.ndimage as ndimage
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from pyntcloud import PyntCloud
import pyvista as pv
import PVGeo as pg

in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

out_dir = "/data/deg_test_output/"

dir = pyd.Directory(in_dir, out_dir)

input_dir = dir.InputDIR()

output_dir = dir.OutputDIR()

new_input_dir = pyd.Directory(out_dir).InputDIR()

print(input_dir)
print(output_dir)

#Mw loss function (units g/mol/day) Mw loss rate: 900 g/mol/day
#crystallinity (Poisson distribution)
#relate Mw to mechanical properties
#use voxel model to assign Mw value and reduce it using the loss function
#in the amorphous regions
#maybe nodal FEA modelling
#assumptions:
##each voxel has equal density

def MwLossRate():
    return 900 #g/mol/day

def AssignMw(led):
    if str.lower(led) == "high":
        return 220000
    elif str.lower(led) == "med":
        return 270000
    elif str.lower(led) == "low":
        return 350000
    else:
        return "unknown LED"

def AssignXc(led):
    # def init_crystallinity(self, chi):
    #     # crystallinity = probability a pixel will be crystalline
    #     bin_prob = np.random.binomial(1, chi)
    #     return 1 if bin_prob == 1 else 0
    bin_prob = np.random.binomial(1, chi)
    return 1 if bin_prob == 1 else 0
    