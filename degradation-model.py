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
import math

in_dir = "/data/sample1/25/uct/tiff/"  # '/data/sample1/25/model/25.stl' #"/data/sample1/25/uct/tiff/"

out_dir = "/data/deg_test_output/"

dir = pyd.Directory(in_dir, out_dir)

input_dir = dir.InputDIR()

output_dir = dir.OutputDIR()

new_input_dir = pyd.Directory(out_dir).InputDIR()

# print(input_dir)
# print(output_dir)

#Fick's Law of Diffusion 
##Assume 1D initially, infinite source
##Molarity (concentration): C = m/V * 1/MW
##m = mass of solute (g), V is volume of solution in (L), MW is molecular weight, C is molar concentration (mol/L)

diff_coeff = {"25": 51.7e-12, "37": 67.6e-12, "50": 165e-12} #x10^(-12) m^2/s
diff_coeff_mm = {"37": 67.6e-5} #mm^2/s
# print(diff_coeff_mm)
pha_density = 1.240 * (1/1000**2) #g/m3
pixel_scale = 1 #mm/px * 1px
voxel_vol = 1 #mm^3
voxel_mass = pha_density * voxel_vol
water_mass = voxel_mass * 0.00984
water_conc = water_mass/voxel_vol

#Fick's 2nd Law determines concentration change over time - eq. similar to heat eq
#x goes from 0 -> 1 going through the length of the voxel
#iter = time, the function is meant to run each iteration to update the concetration of water in the exposed voxel
#conecntration units: mol/mm^3
#concentration at surface = M_s = saturation mass of water
#therefore, Fick function determines 

def Fick(diff, x, t):
    """
    returns C/C0
    C0 = water_conc
    """
    return math.erfc(x/(math.sqrt(4*diff*t)))
    # return (1/(math.sqrt(math.pi*diff*t)))*(math.exp(-((x**2)/(4*diff*t))))

#once C/C0 reaches 0.5, random walk takes place (initially model 1D flow, move water concentration to next pixel)


print(Fick(diff_coeff_mm["37"], 1, 1e30))

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
    