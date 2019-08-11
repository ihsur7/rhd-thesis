import pystruts as pys
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndimage
import porespy as ps

imdir = '/data/downsample-2048-man-thres/'
im = '0-lx'
data = pys.Data().data

im1 = pys.ImageImporter(data, imdir).import_image(im)
# plt.imshow(data['0-lx']['raw_data'], cmap='binary')
im1 = pys.Filters(data, 15).median_filter()

im1 = pys.LocalThickness(data).local_thickness(sizes=25)

im1 = pys.Measure(data).measure_all(voxel_size=1, bins=10, log=False)
print(data)
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