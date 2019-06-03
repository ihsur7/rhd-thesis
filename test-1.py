import numpy as np
import pyassimp
import pyflann
import os
import time
import json

directory = 'data,sample1,25.stl'
# outpath = 'data,sample1,'
directory2 = 'data/sample1/25.stl'
outpath = '../data/sample1/'

#  directory = '../data/sample1/25.stl', outpath = '../data/sample1/'

# scene = pyassimp.load(directory2)

class Directory():
    def __init__(self, input_str):
        self.input_str = input_str
    
    def Output(self):
        if ',' in self.input_str:
            separator = ','
            working_path = os.path.join(*self.input_str.split(separator))
            # print("Working path is: " + working_path)
            # print('test')
            return working_path
        elif '/' in self.input_str:
            separator = '/'
            working_path = os.path.join(*self.input_str.split(separator))
            # print("Working path is: " + working_path)
            return working_path
        else:
            print('Please use a comma or forward-slash as a separator.')
            return
        # path = os.path.join(*self.input_str.split())

newDIR = Directory(directory2).Output()

class Voxelise():
    def __init__(self, directory, outpath, coeff = 1.0, size = (192, 192, 200)):
        self.directory = directory
        self.outpath = outpath
        self.coeff = coeff
        self.size = size

    def LoadMesh(self):
        self.scene = pyassimp.load(self.directory)
        self.mesh_count = len(self.scene.meshes)
        print("Load Complete. Mesh Count: {}".format(self.mesh_count))

    def GetBoundingBox(self, scene):
        if len(self.scene.meshes) == 0:
            print("Scene's meshes attribute has no mesh.")
            return (0,0,0,0,0,0)

        self.mesh_1 = self.scene.meshes[0]
        xmax, ymax, zmax = np.amax(self.mesh_1.vertices, axis=0)
        xmin, ymin, zmin = np.amin(self.mesh_1.vertices, axis=0)

        for index in range(1, len(self.scene.meshes)):
            mesh_t = scene.meshes[index]
            xmax_t, ymax_t, zmax_t = np.amax(mesh_t.vertices, axis=0)
            xmin_t, ymin_t, zmin_t = np.amin(mesh_t.vertices, axis=0)

            if xmax_t > xmax:
                xmax = xmax_t
            if ymax_t > ymax:
                ymax = ymax_t
            if zmax_t > zmax:
                zmax = zmax_t
            if xmin_t > xmin:
                xmin = xmin_t
            if ymin_t > ymin:
                ymin = ymin_t
            if zmin_t > zmin:
                zmin = zmin_t
        
        print("Bounding box: xmax: {0}, ymax: {1}, zmax:{2}, xmin: {3}, ymin: {4}, zmin: {5}".format(xmax, ymax, zmax, xmin, ymin, zmin))
        return (xmax, ymax, zmax, xmin, ymin, zmin)
    
    def MeshVoxel(self, startpoint, edge, mesh, voxel, str = '0'):
        vertices = mesh.vertices

        print("The mesh {0} has vertices {1}".format(str, vertices.shape))

        flann = pyflann.FLANN()
        params = flann.build_index(vertices, algorithm = "kdtree", trees = 4)

        width, height, length = voxel.shape
        start_time = time.time()
        landmark = self.coeff * edge

        for x in range(width):
            for y in range(height):
                for z in range(length):
                    voxel_center = np.array([[
                        startpoint[0] + x * edge,
                        startpoint[1] + y * edge,
                        startpoint[2] + z * edge
                    ]], dtype = np.float32)

                    result, dists = flann.nn_index(voxel_center, 1, checks = params['checks'])
                    index = result[0]
                    vertex = vertices[index,:]
                    distance = np.sqrt(((vertex - voxel_center) ** 2).sum())

                    if distance <= landmark:
                        voxel[x,y,z] = 1

        print("The mesh {0} completed successfully in {1}s".format(str, (time.time()-start_time)))

    def SaveVoxel(self, filename, voxel):
        startPoint = 0
        if filename.rfind("/") != 1:
            startPoint = filename.rfind("/") + 1

        filename = filename[startPoint:filename.rfind('.')]
        # np.save(os.path.join(self.outpath, filename) + ".npy", voxel)

        array = voxel.reshape(-1,)
        json_str = json.dumps(array.tolist())
        json_file = open(os.path.join(self.outpath, filename) + ".json", "w+")
        json_file.truncate()
        json_file.write(json_str)
        json_file.close()
    
    def Voxelisation(self):
        voxel_width = self.size[0]
        voxel_height = self.size[1]
        voxel_length = self.size[2]

        voxel = np.zeros(shape = (voxel_width, voxel_height, voxel_length), dtype = np.int8)

        self.scene = pyassimp.load(self.directory)
        meshcount = len(self.scene.meshes)

        boundingbox = self.GetBoundingBox(self.scene)

        center = np.array([
            (boundingbox[0] + boundingbox[3])/2,
            (boundingbox[1] + boundingbox[4])/2,
            (boundingbox[2] + boundingbox[5])/2
        ])
        x_edge = (boundingbox[0] - boundingbox[3]) / voxel_width
        y_edge = (boundingbox[1] - boundingbox[4]) / voxel_height
        z_edge = (boundingbox[2] - boundingbox[5]) / voxel_length

        edge = max(x_edge, y_edge, z_edge)
        print("x_edge: {0}, y_edge: {1}, z_edge: {2}, edge: {3}".format(x_edge, y_edge, z_edge, edge))

        start = center - np.array([
            voxel_width // 2 * edge,
            voxel_height // 2 * edge,
            voxel_length // 2 * edge,
        ])

        print("center: {0}, start: {1}".format(center, start))

        for index in range(meshcount):
            self.MeshVoxel(start, edge, self.scene.meshes[index], voxel, str(index))
        
        print("Voxelisation completed.")

        self.SaveVoxel("output", voxel)

        return voxel



# mesh = Voxelise(directory, outpath).LoadMesh()

# newdir = Directory(directory2).Output()

newmesh = Voxelise(newDIR, outpath).Voxelisation()