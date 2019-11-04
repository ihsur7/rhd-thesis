import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy.ndimage
import scipy.interpolate

E_array = np.array([[0, 0, 3.072988331, 3.28906711, 3.532306716],
                    [0, 3.047586194, 3.12582417, 3.403719195, 3.597742014],
                    [2.909085043, 3.009320386, 3.303613553, 3.116011416, 3.378904309],
                    [3.044460827, 3.160782701, 3.24340315, 3.220933391, 3.536180829],
                    [0, 3.055505075, 3.203535298, 3.294723439, 3.216243253]])

# x = y = np.arange(5)
# xi, yi = np.meshgrid(x, y)

# E_array = scipy.interpolation.griddata((xi, yi), E_array, method='cubic')

# E_array = scipy.ndimage.zoom(E_array,3)

# print(E_array)

# plt.contourf(E_array)#, alpha=0.5)
# plt.colorbar()
# # plt.savefig('contour-test-2.png', cmap='RdBu')
# plt.show()

# plot_aspect = 1.2
# plot_height = 10.0
# plot_width = int(plot_height*plot_aspect)

# plt.figure(figsize=(plot_width, plot_height), dpi=100)
# plt.subplots_adjust(left=0.10, right=1.00, top=0.90, bottom=0.06, hspace=0.30)
# subplot1 = plt.subplot(111)

# implot = subplot1.imshow(plt.imread(r'W0-NZ-indent-2-color-crop.tif'), interpolation='nearest', alpha=0.5, extent=[-1024, 1024, -1024, 1024])
# pp = plt.contourf(E_array, apha=0.5, antialiased=True)
# subplot1.contourf()
# implot.imshow()
img = plt.imread('W20-NZ-indent-color-crop.tif')

fig, ax = plt.subplots()
ax.imshow(img, extent=[-1.25, 5.25, -1.25, 5.25])
# ax.contourf(E_array, alpha=.5)
plot = ax.contourf(E_array, alpha=0.75, cmap='rainbow')
cb = fig.colorbar(plot)
cb.set_label('Elastic Modulus/E (GPa)')
plt.axis('off')

plt.title('W20-NZ')
plt.show()