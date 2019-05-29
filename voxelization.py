import numpy as np
import stl
import os
import pyflann

directory4 = 'data,sample1,25.stl'
outpath = 'data,sample1,'

class Voxelise():
    def __init__(self, directory, outpath, coeff = 1.0, size = (192, 192, 200)):#, outpath, coeff, size):
        self.directory = directory
        self.coeff = coeff
        self.size = size
        self.outpath = outpath

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
        if len(self.mesh_dims) == 0:
            print('Mesh has no points.')
            return (0,0,0,0,0,0)
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
        
        return self.start, self.edge, self.center, self.bbox, self.voxel_map

    def InitialiseVoxel(self):
        self.vertices = self.LoadMesh().points
        self.inimesh = self.InitialiseMesh()
        self.startpoint = self.inimesh[0]
        print("The mesh has vertices {}".format(self.vertices.shape))

        self.flann = pyflann.FLANN()
        self.params = self.flann.build_index(self.vertices, algorithm = "kdtree", trees = 4)
        print(self.params)

        self.x, self.y, self.z = self.inimesh[4].shape

        self.landmark = self.coeff * self.inimesh[1]

        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    self.voxel_center = np.array([[
                        self.startpoint[0] + i * self.inimesh[1],
                        self.startpoint[1] + j * self.inimesh[1],
                        self.startpoint[2] + k * self.inimesh[1]]], dtype=np.float32)
                    self.result, self.dists = self.flann.nn_index(self.voxel_center, 1, checks = self.params["checks"])
                    self.index = self.result[0]
                    self.vertex = self.vertices[self.index,:]
                    self.distance = np.sqrt(((self.vertex - self.voxel_center) ** 2).sum())

                    if self.distance <= self.landmark:
                        self.inimesh[4][i,j,k] = 1
        
        return self.inimesh[4]
        
    def VoxeliseMesh(self):

        for l in range(len(self.LoadMesh().points)):
            self.InitialiseVoxel()
        
        return self.inimesh
    
    def SaveVoxel(self, filename):
        self.voxelise = self.VoxeliseMesh()
        self.startPoint = 0
        if self.filename.rfind("/") != -1:
            self.startPoint = filename.rfind("/") + 1

        self.filename = self.filename[self.startPoint:filename.rfind('.')]
        np.save(os.path.join(self.outpath, self.filename) + ".npy", self.VoxeliseMesh())

            

        

newmesh = Voxelise(directory4,outpath).VoxeliseMesh()
