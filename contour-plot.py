import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

E_array = np.array([[3.11074799, 3.231612906, 3.49612078, 3.395818561, 3.574984294],
                    [3.101399746, 3.593840932, 3.583519062, 3.432509393, 3.493042067],
                    [3.106757791, 3.295204051, 3.417707209, 3.507972937, 3.819293868],
                    [3.157631006, 3.498358303, 3.493396481, 3.505547483, 3.823644558],
                    [3.603378009, 3.629423336, 3.800265289, 3.885838633, 3.479162909]])

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
img = plt.imread('W0-NZ-indent-2-color-crop.jpg')

fig, ax = plt.subplots()
ax.imshow(img, interpolation='none', extent=[-1.5, 5.5, -1.5, 5.5])
# ax.contourf(E_array, alpha=.5)
cb = fig.colorbar(ax.contourf(E_array, alpha=0.5, cmap='Reds'))
cb.set_label('Elastic Modulus/E (MPa)')
plt.title('W0-NZ')
plt.show()