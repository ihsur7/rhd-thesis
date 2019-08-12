import pystruts as pys
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndimage
import porespy as ps
import os

imdir = '/data/downsample-2048-man-thres/'


for i in os.listdir(os.path.join(imdir)):
    print(i)

im = '0-lx'
data = pys.Data().data
data2 = pys.Data().data

im1 = pys.ImageImporter(data, imdir).import_image(im)
# plt.imshow(data['0-lx']['raw_data'], cmap='binary')
im1 = pys.Filters(data, 15).median_filter()

pspyimage = pys.ImageImporter(data2, imdir).import_image(im)
pspyimage = pys.Filters(data2, 15).median_filter()

print(data2)

sizes = np.arange(start=1, stop=200, step=0.1)
print(len(np.ndarray.tolist(sizes)))
print(sizes)
im1 = pys.LocalThickness(data).local_thickness(sizes=sizes)

im1 = pys.Measure(data).measure_all(voxel_size=3.32967, bins=200, log=False) #um/px


im2 = pys.save_csv(data)

# pspy = ps.filters.local_thickness(data2[im]['filter'], sizes=sizes, mode='dt')
# pspy = ps.metrics.pore_size_distribution(pspy, bins=100, log=False, voxel_size=3.32967)
# plt.bar(x = pspy.R, height=pspy.pdf, width=pspy.bin_widths, edgecolor='k', linewidth=2)
# plt.show()
# plt.bar(x=data[im]['R'], height=data[im]['pdf'], width=data[im]['bin_widths'], 
# edgecolor='k', linewidth=2)
# plt.show()

# plt.show()

# im2 = np.array([[0, 0, 1, 1],
#                 [0, 0, 1, 1],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 0]])

# im2 = morph.skeletonize(im2)
# loc = np.transpose(np.nonzero(im2))
# dt = ndimage.distance_transform_edt(im2)
# imresults = np.zeros(shape=im2.shape)
# for i,j in loc:
#     print(i, j)
#     imresults[i,j] = dt[i,j]
# print(imresults)