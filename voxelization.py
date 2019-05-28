import numpy as np
import stl
import os

class Voxelise():
    def __init__(self, directory):#, outpath, coeff, size):
        self.directory = directory
        # self.outpath = outpath
        # self.coeff = coeff
        # self.size = size

    def Initialise(self):
        self.voxel_w = self.size[0]
        self.voxel_h = self.size[1]
        self.voxel_l = self.size[2]

        self.voxel_map = np.zeros(shape=(self.voxel_w, self.voxel_h, self.voxel_l), dtype=np.int8)

    def LoadMesh(self):
        self.mesh = stl.Mesh.from_file(os.path.join(*self.directory.split(',')))
        # print(self.mesh.points)
        print(len(self.mesh.points))
        print(self.mesh.points[0])