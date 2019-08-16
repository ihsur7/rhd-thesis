import pystruts as pys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndimage
import porespy as ps
import os
import pydirectory as pyd
import multiprocessing as multi

imdir = '/data/downsample-2048-man-thres/'
outdir = '/data/freq-lt-pores/'
direct = pyd.Directory(imdir, outdir)
work_dir = direct.InputDIR()
out_dir = direct.OutputDIR()
sizes = np.logspace(start=np.log10(200), stop=0, num=100)[-1::-1]
print(sizes) #25 #np.arange(start=1, stop=500, step=0.1)
bins = np.linspace(start=1, stop=601, num=300)

for i in os.listdir(work_dir):
    i = i[:-4]
    print('now processing... ' + i)
    data = pys.Data().data
    im1 = pys.ImageImporter(data, imdir).import_image(i)
    im1 = pys.Filters(data, 15).median_filter()
    im1 = pys.LocalThickness(data).local_thickness(sizes, invert=True)
    # grid = gs.GridSpec(1, 3)
    # plt.figure()
    # ax = plt.subplot(grid[0, 0])
    # plt.imshow(data[i]['raw_data'])
    # plt.axis('off')
    # plt.title('raw_data')

    # ax = plt.subplot(grid[0, 1])
    # plt.imshow(data[i]['filter'])
    # plt.axis('off')
    # plt.title('filter')

    # ax = plt.subplot(grid[0, 2])
    # plt.imshow(data[i]['lt'])
    # plt.axis('off')
    # plt.title('lt')

    # plt.suptitle(i)
    # plt.savefig(out_dir+i+'.png', dpi=300)
    # plt.close()
    # # im1 = pys.Measure(data).measure_all(voxel_size=3.32967, bins=bins, log=False)
    # im2 = pys.save_csv(data, out_dir)
    # print('saved... ' + i + '.csv')
    # im2 = pys.save_lt(data, out_dir)
    # im2 = pys.save_freq_count(data, out_dir)
    print(pys.hist(pys._parse_lt(data)))
    print('saved... '+i+'.csv')

# im = '0-lx'
# data = pys.Data().data
# data2 = pys.Data().data

# im1 = pys.ImageImporter(data, imdir).import_image(im)
# # plt.imshow(data['0-lx']['raw_data'], cmap='binary')
# im1 = pys.Filters(data, 15).median_filter()

# pspyimage = pys.ImageImporter(data2, imdir).import_image(im)
# pspyimage = pys.Filters(data2, 15).median_filter()

# print(data2)

# sizes = np.arange(start=1, stop=200, step=0.01)
# print(len(np.ndarray.tolist(sizes)))
# print(sizes)
# im1 = pys.LocalThickness(data).local_thickness(sizes=sizes)

# im1 = pys.Measure(data).measure_all(voxel_size=3.32967, bins=200, log=False) #um/px


# im2 = pys.save_csv(data)

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