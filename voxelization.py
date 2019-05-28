import numpy as np
import stl
import os

directory4 = 'data,sample1,25.stl'

class Voxelise():
    def __init__(self, directory, size = (192, 192, 200)):#, outpath, coeff, size):
        self.directory = directory
        # self.outpath = outpath
        # self.coeff = coeff
        self.size = size

    def LoadMesh(self):
        self.mesh = stl.Mesh.from_file(os.path.join(*self.directory.split(',')))
        # print(self.mesh.points)
        self.mesh_count = len(self.mesh.points)
        # print(len(self.mesh.points))
        # print(self.mesh.points[0])
        # print(self.mesh.max_)
        # print(self.mesh.get_mass_properties())

        return self.mesh

    def GetBoundingBox(self):
        self.xmax = self.ymax = self.zmax = self.xmin = self.ymin = self.zmin = None
        self.mesh_dims = self.LoadMesh().points
        for p in self.mesh_dims:
            if self.xmin is None:
                self.xmin = p[stl.Dimension.X]
                self.xmax = p[stl.Dimension.X]
                self.ymin = p[stl.Dimension.Y]
                self.ymax = p[stl.Dimension.Y]
                self.zmin = p[stl.Dimension.Z]
                self.zmax = p[stl.Dimension.Z]
            else:
                self.xmax = max(p[stl.Dimension.X], self.xmax)
                self.xmin = min(p[stl.Dimension.X], self.xmin)
                self.ymax = max(p[stl.Dimension.Y], self.ymax)
                self.ymin = min(p[stl.Dimension.Y], self.ymin)
                self.zmax = max(p[stl.Dimension.Z], self.zmax)
                self.zmin = min(p[stl.Dimension.Z], self.zmin)
        print(self.xmax, self.ymax, self.zmax, self.xmin, self.ymin, self.zmin)

        return(self.xmax, self.ymax, self.zmax, self.xmin, self.ymin, self.zmin)

    def InitialiseMesh(self):
        self.voxel_x = self.size[0]
        self.voxel_y = self.size[1]
        self.voxel_z = self.size[2]

        self.voxel_map = np.zeros(shape=(self.voxel_x, self.voxel_y, self.voxel_z), dtype=np.int8)

        self.bbox = self.GetBoundingBox()
        self.center = np.array([(self.bbox[0] + self.bbox[3])/2,
                                (self.bbox[1] + self.bbox[4])/2,
                                (self.bbox[2] + self.bbox[5])/2])

        self.x_edge = (self.bbox[0] - self.bbox[3]) / self.voxel_x
        self.y_edge = (self.bbox[1] - self.bbox[4]) / self.voxel_y
        self.z_edge = (self.bbox[2] - self.bbox[5]) / self.voxel_z
        
        self.edge = max(self.x_edge, self.y_edge, self.z_edge)
        print(self.center)
        print ("x_edge: {0}, y_edge: {1}, z_edge: {2}, edge: {3}".format(
        self.x_edge, self.y_edge, self.z_edge, self.edge))

        self.start = self.center - np.array([self.voxel_x // 2 * self.edge,
                                             self.voxel_y // 2 * self.edge,
                                             self.voxel_z // 2 * self.edge])

        print("center: {0}, start: {1}".format(self.center, self.start))

        for index in range(self.mesh_count):
            pass

newmesh = Voxelise(directory4).InitialiseMesh()
