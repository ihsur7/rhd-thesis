import numpy as np
import stl
import os
import pyflann
import pyassimp

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
 
def voxelise(filename,\
             npoutpath = '../voxel-numpy/',\
             jsonoutpath = '../voxel-json',\
             coeff = 1.0, size = (192, 192, 200)):
    voxel_width = size[0]
    voxel_height = size[1]
    voxel_length = size[2]

    voxel = np.zeros(shape = (voxel_width, voxel_height, voxel_length), dtype = np.int8)

    mesh = stl.Mesh.from_file(os.path.join(*directory.split(',')))
    meshcount = len(mesh.points)

    boundingbox = _getBoundingBox(mesh)

    center = np.array([(boundingbox[0] + boundingbox[3])/2,
                       (boundingbox[1] + boundingbox[4])/2,
                       (boundingbox[2] + boundingbox[5])/2])

    x_edge = (boundingbox[0] - boundingbox[3]) / voxel_width
    y_edge = (boundingbox[1] - boundingbox[4]) / voxel_height
    z_edge = (boundingbox[2] - boundingbox[5]) / voxel_length

    edge = max(x_edge, y_edge, z_edge)

    print("x_edge: {0}, y_edge: {1}, z_edge: {2}, edge: {3}".format(x_edge, y_edge, z_edge, edge))

    start = center - np.array([voxel_width // 2 * edge,
                               voxel_height // 2 * edge,
                               voxel_length // 2 * edge])
    
    print("center: {0}, start: {1}".format(center, start))

    for index in range(meshcount):
        _meshVoxel(start, edge, mesh, voxel, coeff, str(index))
    print("mesh voxelised")

def _getBoundingBox(mesh):
    xmax = ymax = zmax = xmin = ymin = zmin = None
        mesh_dims = mesh.points
        if len(mesh_dims) == 0:
            print('Mesh has no points.')
            return (0,0,0,0,0,0)
        for p in mesh_dims:
            if xmin is None:
                xmin = p[stl.Dimension.X]
                xmax = p[stl.Dimension.X]
                ymin = p[stl.Dimension.Y]
                ymax = p[stl.Dimension.Y]
                zmin = p[stl.Dimension.Z]
                zmax = p[stl.Dimension.Z]
            else:
                xmax = max(p[stl.Dimension.X], xmax)
                xmin = min(p[stl.Dimension.X], xmin)
                ymax = max(p[stl.Dimension.Y], ymax)
                ymin = min(p[stl.Dimension.Y], ymin)
                zmax = max(p[stl.Dimension.Z], zmax)
                zmin = min(p[stl.Dimension.Z], zmin)
        print(xmax, ymax, zmax, xmin, ymin, zmin)

        return(xmax, ymax, zmax, xmin, ymin, zmin)

def _meshVoxel(startpoint, edge, mesh, voxel, coeff = 1.0, str = "0"):


newmesh = Voxelise(directory4,outpath).VoxeliseMesh()
