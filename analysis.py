import porespy as ps
from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pydirectory as dct
import os

week = [0, 4, 8, 12, 16, 20]
direction = ['l', 'n']
sample = ['x', 'y', 'z']



def load_data(folder, data, resize = None):
    data['input'] = {}
    for item in os.listdir(folder):
        # print(item)
        if not resize:
            im = np.array(Image.open(folder+item).convert('L'))
            # print(data)
            data['input'][str(item)] = im
        else:
            im = np.array(PIL.ImageOps.fit(Image.open(folder+item).convert('L'), size=tuple(resize)))
            data['input'][str(item)] = im
    # print(data['input'])#['20-lx.tif'])
    # print(data)
    return data


def apply_filter(data):
    data['lt'] = {}
    for item, value in data['input'].items():
        print('\nprocessing...'+item)
        data['lt'][str(item)] = ps.filters.local_thickness(value)
        print('done')

    return data

def analyse(data):
    data['psd'] = {}
    vox = 2048.0*774.0/5278.0
    for item, value in data['lt'].items():
        data['psd'][str(item)] = ps.metrics.pore_size_distribution(value, log=False, voxel_size=1.0/vox)

    return data

def plot(outfolder, data):
    # 0-x
    # itemlist = ['20-lx', '20-nx']
    itemlist1 = [[['0-lx', '0-nx'], ['0-ly', '0-ny'], ['0-lz', '0-nz']],
                 [['4-lx', '4-nx'], ['4-ly', '4-ny'], ['4-lz', '4-nz']],
                 [['8-lx', '8-nx'], ['8-ly', '8-ny'], ['8-lz', '8-nz']],
                 [['12-lx', '12-nx'], ['12-ly', '12-ny'], ['12-lz', '12-nz']],
                 [['16-lx', '16-nx'], ['16-ly', '16-ny'], ['16-lz', '16-nz']],
                 [['20-lx', '20-nx'], ['20-ly', '20-ny'], ['20-lz', '20-nz']]]
    itemlist2 = [[['0-lx', '4-lx', '8-lx', '12-lx', '16-lx', '20-lx'], ['0-nx', '4-nx', '8-nx', '12-nx', '16-nx', '20-nx']],
                 [['0-ly', '4-ly', '8-ly', '12-ly', '16-ly', '20-ly'], ['0-ny', '4-ny', '8-ny', '12-ny', '16-ny', '20-ny']],
                 [['0-lz', '4-lz', '8-lz', '12-lz', '16-lz', '20-lz'], ['0-nz', '4-nz', '8-nz', '12-nz', '16-nz', '20-nz']]]
    # for i in itemlist1:
    #     for j in i:
    #         plt.figure()
    #         line1 = data['psd'][j[0]+'.tif']
    #         line2 = data['psd'][j[1]+'.tif']
    #         plt.plot(line1.R, line1.cdf, label='L')
    #         plt.plot(line2.R, line2.cdf, label='N')
    #         plt.xlabel('invasion size [mm]')
    #         plt.ylabel('volume fraction invaded')
    #         plt.legend()
    #         plt.title(j)
    #         plt.savefig(outfolder+j[0]+j[-1]+'.png', dpi=300)
    for k in itemlist2:
        for l in k:
            plt.figure()
            line1 = data['psd'][l[0]+'.tif']
            line2 = data['psd'][l[1]+'.tif']
            line3 = data['psd'][l[2]+'.tif']
            line4 = data['psd'][l[3]+'.tif']
            line5 = data['psd'][l[4]+'.tif']
            line6 = data['psd'][l[5]+'.tif']
            plt.plot(line1.R, line1.cdf, label='W0')
            plt.plot(line2.R, line2.cdf, label='W4')
            plt.plot(line3.R, line3.cdf, label='W8')
            plt.plot(line4.R, line4.cdf, label='W12')
            plt.plot(line5.R, line5.cdf, label='W16')
            plt.plot(line6.R, line6.cdf, label='W20')
            plt.xlabel('invasion size [mm]')
            plt.ylabel('volume fraction invaded')
            plt.legend()
            plt.title(l)
            plt.savefig(outfolder+l[0]+l[-1]+'.png', dpi=300)
    
    return


if __name__ == "__main__":
    folder = '/data/downsample-2048-man-thres/'
    outfolder = '/data/output/'
    workdir = dct.Directory(folder,outfolder)
    data = {}
    input_files = load_data(workdir.InputDIR(), data)#, resize=[500, 500])
    filtered = apply_filter(input_files)
    analysed = analyse(filtered)
    plot(workdir.OutputDIR(), analysed)
    # print(data['lt'])
    # print(data['psd'])

#data = {'input':{'0-lx.tif':[array], '0-ly.tif':[array]}, 'lt':{etc}}


  # line1 = data['psd'][itemlist[0]+'.tif']
    # line2 = data['psd'][itemlist[1]+'.tif']
    # plt.plot(line1.R, line1.cdf, label='L')
    # plt.plot(line2.R, line2.cdf, label='N')
    # plt.xlabel('invasion size [voxels]')
    # plt.ylabel('volume fraction invaded [voxels]')
    # plt.legend()
    # plt.title('Week '+itemlist[0][0]+' - Sample '+itemlist[0][-1])
    # plt.savefig(outfolder+itemlist[0][0]+itemlist[0][-1]+'.png', dpi=300)
    
    # for i in np.arange(0, 24, 4):
    #     for j in ['x', 'y', 'z']:
    #         line1 = data['psd'][str(i)+'-'+'l'+str(j)+'.tif']
    #         line2 = data['psd'][str(i)+'-'+'n'+str(j)+'.tif']
    #         plt.plot(line1.R, line1.cdf, label='L')
    #         plt.plot(line2.R, line2.cdf, label='N')
    #         plt.xlabel('invasion size [voxels]')
    #         plt.ylabel('volume fraction invaded [voxels]')
    #         plt.legend()
    #         plt.title(str(i)+' '+str(j))
    #         plt.savefig(outfolder+str(i)+str(j)+'.png', dpi=300)
    

    # plt.plot(line1.R, line1.cdf, label='L')
    # plt.plot(line2.R, line2.cdf, label='N')
    # plt.xlabel('invasion size [voxels]')
    # plt.ylabel('volume fraction invaded [voxels]')
    # plt.legend()
    # plt.title(title)
    # plt.savefig(outfolder+title+'.png', dpi=300)
    # for i, key in enumerate(data['psd'].keys()):
        

    # fig1 = plt.plot
    # for x, y in data['psd'].items():
    #     fig = plt.plot(y.R, y.cdf, 'bo-')
    #     plt.xlabel('invasion size [voxels]')
    #     plt.ylabel('volume fraction invaded [voxels]')
    #     plt.title(x)
    #     plt.savefig(outfolder+x.replace('.tif', '.png'), dpi=300)
    # fig = plt.plot(data['psd']['0-lx.tif'].