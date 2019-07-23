import scipy as sp
import numpy as np
import scipy.ndimage
import skimage.morphology as skim
import matplotlib.pyplot as plt

size = (10,10)

im = np.random.randint(0, 2, size=size)

dt = sp.ndimage.distance_transform_edt(im > 0)
print(dt)

def getedge(shape, thickness=1, return_indices=False):
        t = thickness
        border = sp.ones(shape, dtype=bool)
        print(border)
        border[t:-t, t:-t] = False
        print(border)
        if return_indices:
            border = sp.where(border)
        return border

def getcircle(radius):
    return skim.disk(radius, dtype=bool)

# a = getedge(im.shape)
# b = sp.where(a)
# print(b)

a = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=5)
# print(a)
imresults = np.zeros(np.shape(im))
for i in a:
    imtemp = dt >= i
    print(imtemp)
    print(~imtemp)
    if sp.any(imtemp):
        imtemp = sp.ndimage.distance_transform_edt(~imtemp) < i
        # print(imtemp)
        imresults[(imresults == 0)*imtemp] = i

# print(imresults)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[5,5], constrained_layout=True)
for i,j in zip(ax.flat, [im, dt, imresults]):
    im2 = i.imshow(j)
fig.colorbar(im2, ax=ax.flat)

# plt.tight_layout(True)
plt.show()

# print(dt)