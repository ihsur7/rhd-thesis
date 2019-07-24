import porespy as spy
from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import pydirectory as dct

filestring = "/data/downsample-2048-man-thres/0-ly.tif"

week = '0'
direction = ['l', 'n']
sample = ['x', 'y', 'z']

folder = "/data/downsample-2048-man-thres/"
# #list all files in the folder -> add files to a list/dictionary -> run analysis -> save plots
# # def listdir_fullpath(pathname):
# #     print(os.listdir(pathname))
# #     return [os.path.join(pathname, f) for f in os.listdir(pathname)]

# workdir = dct.Directory(folder, folder)
# data = {}
# for item in os.listdir(workdir.InputDIR()):
#     im = np.array(Image.open(os.path.join(workdir.InputDIR()+item)).convert("L"))
#     data[str(item)] = im

# print(data)
    
# spydata = {}
# spymetric = {}
# for items, values in data.items():
#     spydata[items] = spy.filters.local_thickness(values)

# for items, values in spydata.items():
#     spymetric[items] = spy.metrics.pore_size_distribution(values)
#     fig = plt.plot(spymetric[items].R, spymetric[items].cdf, 'bo-')
#     plt.xlabel("invasion size [voxels]")
#     plt.ylabel("volume fraction invaded [voxels]")
#     plt.show()




# f1 = folder + "0-lx.tif"
# f2 = folder + "4-lx.tif"
# f3 = folder + "8-lx.tif"
# f4 = folder + "12-lx.tif"
# f5 = folder + "16-lx.tif"
# f6 = folder + "20-lx.tif"

# print(f1)

workdir = dct.Directory(filestring, filestring)

im = Image.open(workdir.InputDIR()).convert("L")
# iminvert = PIL.ImageOps.invert(im)
imarray = np.array(im)

lt = spy.filters.local_thickness(imarray)
plt.imshow(lt)
plt.show()

data = spy.metrics.pore_size_distribution(lt, log=False)
fig = plt.plot(data.R, data.cdf, 'bo-')
plt.xlabel('invasion size [voxels]')
plt.ylabel('volume fraction invaded [voxels]')
plt.show()