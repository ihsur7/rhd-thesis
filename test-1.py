import numpy as np
import pyassimp
import pyflann
import os

directory = 'data,sample1,25.stl'
outpath = 'data,sample1,'

#  directory = '../data/sample1/25.stl', outpath = '../data/sample1/'

class Voxelise():
    def __init__(self, directory, outpath, coeff = 1.0, size = (192, 192, 200)):
        self.directory = directory
        self.outpath = outpath
        self.coeff = coeff
        self.size = size

    def LoadMesh(self):
        self.scene = pyassimp.load(os.path.join(*self.directory.split(',')))
        self.mesh_count = len(self.scene.meshes)
        print("Load Complete. Mesh Count: {}".format(self.mesh_count))

mesh = Voxelise(directory, outpath).LoadMesh()