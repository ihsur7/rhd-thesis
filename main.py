import imageprocessor as IP
# import pyflann
import numpy as np
import stl as mesh
import os
import pystruts as pys
# import pyassimp

import voxelization as vox
import pydirectory as direct

# directory = 'data,sample1,25,*.bmp'
# directory2 = 'data,sample1,25,'
# directory3 = '/data/sample1/'
# directory4 = '/data/sample1/25.stl'

indir1 = '/data/sample1/25/uct/bmp/*.bmp'
indir2 = '/data/sample1/25/model/25.stl'
outdir1 = '/data/sample1/25/model/output/'

working_dir = direct.Directory(indir2,outdir1)
# print(inputdir.OutputDIR(), inputdir.InputDIR())
# newmesh = vox.Voxelise(inputdir.InputDIR(), inputdir.OutputDIR())

newmesh = vox.Voxelise(working_dir.InputDIR(), working_dir.OutputDIR())

# # newmesh.Voxelisation()

# filename = filename[0:filename.rfind('.')]
# print(filename)
