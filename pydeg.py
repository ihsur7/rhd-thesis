from PIL import Image
import PIL.ImageOps as imops
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import skimage.data as skdata
import skimage.io as skio
from skimage import img_as_uint
from pystruts import ImageImporter, Filters
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random
import pydirectory as pyd
import matplotlib.animation
#take edt
#identify pixels on the edge
#turn them off

#for expansion
#invert image
#identify pixels on edge
#turn them on

mpl.use('Agg')
mpl.rcParams['animation.ffmpeg_path'] = 'C:/Program Files/ImageMagick-7.0.8-Q16/ffmpeg.exe'
mpl.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe'
d = pyd.Directory('/data/final_images/', '/data/deg_test/')

in_dir = d.InputDIR()
out_dir = d.OutputDIR()

def iter(n):
    init_img = skio.imread(in_dir+'0-lx.tif', as_gray=True)
    skio.imsave(out_dir+'iter-0.png', init_img)
    i = 1
    imglist = []
    imglist.append(init_img)
    while i < n:
        image = skio.imread(out_dir+'iter-{}.png'.format(i-1), as_gray=True)
        if i < n/20:
            dt = ndimage.distance_transform_edt(~image)
        else:
            dt = ndimage.distance_transform_edt(image)
        imtemp = dt == 1
        if np.any(imtemp):
            edge = np.where(imtemp)
            dec = decline(0.01, 0.2, i)
            randlist_size = math.floor(random.uniform(dec[0], dec[1])*len(edge[0])) #[0.05, 0.2], [0.2, 0.5]
            index_list = np.arange(0, len(edge[0]))
            new_index_list = shuffled(index_list)[0:randlist_size-1]
            col0 = [x for x in edge[0][new_index_list]]
            col1 = [y for y in edge[1][new_index_list]]

            for l, m in zip(col0, col1):
                if i < n/20:
                    image[l, m] = 255
                else:
                    image[l, m] = 0

            skio.imsave(out_dir+'iter-{}.png'.format(i), image)
            imglist.append(image)
        else:
            break
        i += 1

    return imglist

def decline(a, b, i):
    return (a-((a/i)**i), b-((b/i)**i))


def shuffled(a):
    np.random.shuffle(a)
    return a

fig = plt.figure()
imgl = iter(150)
img = plt.imshow(imgl[0], cmap='gray')

def updatefig(j):
    img.set_array(imgl[j])
    return [img]

ani = matplotlib.animation.FuncAnimation(fig, updatefig, frames=range(150), interval=100, blit=True, repeat=True)
ani.save(out_dir+'anim.gif', writer='imagemagick', fps=10, bitrate=-1)
# plt.show()
