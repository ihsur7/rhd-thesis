import imageprocessor as IP
# import pyflann
import numpy as np
import stl as mesh
import os
# import pyassimp

import voxelization as vox
import pydirectory as direct

directory = 'data,sample1,25,*.bmp'
directory2 = 'data,sample1,25,'
directory3 = '/data/sample1/'
directory4 = '/data/sample1/25.stl'

inputdir = direct.Directory(directory4,directory3)
print(inputdir.OutputDIR(), inputdir.InputDIR())
newmesh = vox.Voxelise(inputdir.InputDIR(), inputdir.OutputDIR())

newmesh.Voxelisation()
