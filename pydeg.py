from PIL import Image
import PIL.ImageOps as imops
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import skimage.data as skdata
import skimage.io as skio
from skimage import img_as_uint
from pystruts import ImageImporter, Filters, LocalThickness
import pystruts as pys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import math
import random
import pydirectory as pyd
import matplotlib.animation
import os

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

out_dir2 = pyd.Directory('data/deg_test/', 'data/deg_test2/').OutputDIR()

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
            size = (np.shape(imtemp)[0]-1, np.shape(imtemp)[1]-1)
            for l, m in zip(col0, col1):
                if i < n/20:
                    image[l, m] = 255
                else:
                    image[l, m] = 0
            dt2 = ndimage.distance_transform_edt(image)
            imtemp2 = dt2 == 1
            if np.any(imtemp2):
                edge2 = np.where(imtemp2)
                col3 = [x for x in edge2[0]]
                col4 = [y for y in edge2[1]]
                for e, f in zip(col3, col4):
                    n_list = neighbours(e, f, size)
                    n_list_value = [image[u,v] == 0 for u,v in n_list]
                    result = all(n_list_value)
                    if result:
                        image[e, f] = 0
            skio.imsave(out_dir+'iter-{}.png'.format(i), image)
            imglist.append(image)
            print('img {} saved'.format(i))
        else:
            break
        i += 1

    return imglist

def neighbours(x, y, res):
    #take coordinates, calculate surrounding coordinates -> return a list of coordinates
    #create a list of x coordinates
    x_range = np.arange(x-1, x+2)
    y_range = np.arange(y-1, y+2)
    n_list = []
    for i in x_range:
        for j in y_range: #check if coordinates are negative or larger than image
            if -1 < x <= res[0] and -1 < y <= res[1]:
                if (x != i or y != j) and (0 <= i <= res[0]) and (0 <= j <= res[1]):
                    n_list.append([i, j])
    return n_list

def decline(a, b, i):
    return (a-((a/i)**i), b-((b/i)**i))


def shuffled(a):
    np.random.shuffle(a)
    return a

# fig = plt.figure()
# imgl = iter(150)
# img = plt.imshow(imgl[0], cmap='gray')

# def updatefig(j):
#     img.set_array(imgl[j])
#     return [img]

# ani = matplotlib.animation.FuncAnimation(fig, updatefig, frames=range(len(imgl)), interval=100, blit=True, repeat=True)
# ani.save(out_dir+'anim.gif', writer='imagemagick', fps=10, bitrate=-1)
# plt.show()

filelist = os.listdir(out_dir)
filelist.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

sizes = np.logspace(start=np.log10(200), stop=0, num=100)[-1::-1]
# print(sizes) #25 #np.arange(start=1, stop=500, step=0.1)
bins = np.linspace(start=1, stop=601, num=300)

for i in filelist:
    i = i[:-4]
    file = out_dir + i + '.png'
    data = pys.Data().data
    im1 = pys.ImageImporter(data, out_dir).import_image(i)
    im1 = pys.Filters(data, 3).none() #original 15
    im1 = pys.LocalThickness(data).local_thickness(sizes, invert=True)
    grid = gs.GridSpec(1, 3)
    plt.figure()
    ax = plt.subplot(grid[0, 0])
    plt.imshow(data[i]['raw_data'], cmap='binary')
    plt.axis('off')
    plt.title('raw_data')

    ax = plt.subplot(grid[0, 1])
    plt.imshow(data[i]['filter'], cmap='binary')
    plt.axis('off')
    plt.title('filter')

    ax = plt.subplot(grid[0, 2])
    plt.imshow(data[i]['lt'])
    plt.axis('off')
    plt.title('lt')

    plt.suptitle(i)
    # plt.show()
    plt.savefig(out_dir2+i+'.png', dpi=300)
    plt.close()
    im2 = pys.save_lt(data, out_dir2)

ltdir = pyd.Directory('/data/deg_test2/', '/data/deg_test2output/')
ltlist = os.listdir(ltdir.OutputDIR())