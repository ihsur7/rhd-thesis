import os
import sys
import matplotlib.pyplot as plt
import glob
import vtk
from vtk.util import numpy_support
from vtk.util.misc import vtkGetDataRoot
import numpy as np
from PIL import Image
import nibabel as nib
import plotly.plotly as py
import plotly.graph_objs as go
import warnings

from pyevtk.hl import gridToVTK

# py.tools.set_credentials_file(username='abrafcukincadabra', api_key='B0bzzqaiK7bXw4c1zaVZ')
# warnings.simplefilter(action='ignore', category=FutureWarning)

print(vtk.vtkVersion.GetVTKSourceVersion())

# def plotHeatmap(array, name="plot"):
#     trace = go.Heatmap(z=array)
#     data = Data([
#         Heatmap(
#             z=array,
#             # scl='Greys'
#         )
#     ])
#     layout = Layout(
#         autosize=False,
#         title=name
#     )
#     fig = Figure(data=data, layout=layout)

#     return py.plotly.iplot(fig, filename=name)

class ImageProcessor():
    def __init__(self, directory):
        self.directory = directory
        self.img_list = []

    def ImageImport(self):
        for imgfile in glob.glob(os.path.join(*self.directory.split(','))):
            img = Image.open(imgfile)
            self.img_list.append(img)
        return self.img_list
    
    def CreateVolume(self):
        temp = self.img_list[0]
        arr = np.array(temp) #Another way of converting list to array
        w,d = temp.size
        h = len(self.img_list)
        volume = np.zeros((w,d,h),dtype=np.uint16)
        imgslice = 0
        for im in self.img_list:
            imdata = list(im.getdata())
            n = w
            imdata1 = [imdata[i:i+n] for i in range(0, len(imdata), n)]
            for i,x in enumerate(imdata1):
                for j,y in enumerate(x):
                    volume[i][j][imgslice] = y
            imgslice += 1
        return volume
    def CreateNifti(self):
        self.shape = self.CreateVolume().shape[2]
        self.niftiData = nib.Nifti1Image(self.CreateVolume(), affine=np.eye(4))
        return self.niftiData

    def ShowImage(self, slice = 0):
        array = self.CreateVolume().astype('uint8')
        img = Image.fromarray(array[:, :, slice], 'L')
        img.show()

class ImageProcessor2():
    def __init__(self, directory):
        self.directory = directory
        self.vtkdataroot = vtkGetDataRoot()
        self.img_list = []
        self.colors = vtk.vtkNamedColors()


    def BMPImageReader(self):
        self.reader = vtk.vtkBMPReader()
        self.reader.SetFilePrefix(os.path.join(*self.directory.split(',')))
        self.reader.SetFilePattern('%s25_cropped%03d.bmp')
        # print(self.reader.GetDataExtent())
        self.dims = self.GetDimensions()
        self.reader.SetDataExtent(0, self.dims[2]-1, 0, self.dims[1]-1, 0, self.dims[0]-1)
        # self.reader.SetSpacing(0, 0, 0)

        self.reader.SetDataScalarTypeToUnsignedChar()
        self.reader.SetNumberOfScalarComponents(1)
        # self.reader.SetDirectoryName(os.path.join(*self.directory.split(',')))
        # print(self.reader)
        # self.reader.Update()
        # print(self.reader.GetOutput())
        return self.reader

    def GetDimensions(self):
        self.imageformat = ',*.bmp'
        self.fulldir = self.directory + self.imageformat
        for imgfile in glob.glob(os.path.join(*self.fulldir.split(','))):
            img = Image.open(imgfile)
            self.img_list.append(img)
        temp = self.img_list[0]
        arr = np.array(temp) #Another way of converting list to array
        w,d = temp.size
        h = len(self.img_list)
        return (w, d, h)
    
    def Color(self):
        self.opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self.colorTransferFunction = vtk.vtkColorTransferFunction()

        self.colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
        self.colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

        self.opacityTransferFunction.AddPoint(0, 10.0)
        self.opacityTransferFunction.AddPoint(100, 100.0)

        return self.opacityTransferFunction, self.colorTransferFunction

    def Volume(self):
        self.volume = vtk.vtkVolume()
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()

        self.volumeProperty.SetColor(self.Color()[1])
        self.volumeProperty.SetScalarOpacity(self.Color()[0])
        # self.volumeProperty.ShadeOn()

        self.volumeMapper.SetInputConnection(self.BMPImageReader().GetOutputPort())
        # self.volumeMapper.SetMaximumImageSampleDistance(0.1)

        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)       
        print(self.volume) 
        return self.volume

    def Threshold(self):
        self.threshold = vtk.vtkImageThreshold()
        
        self.threshold.SetInputConnection(self.DataImport().GetOutputPort())

        self.threshold.ThresholdByLower(50)
        self.threshold.ReplaceInOn()
        self.threshold.SetInValue(0)  # set all values below 400 to 0
        self.threshold.ReplaceOutOn()
        self.threshold.SetOutValue(1)  # set all values above 400 to 1
        # self.threshold.Update()
        # print(self.threshold)
        return self.threshold

    def InitialiseStack(self, voltype='nc'):
        self.renderer = vtk.vtkRenderer()
        self.renderWin = vtk.vtkRenderWindow()
        self.renderInteractor = vtk.vtkRenderWindowInteractor()

        self.renderWin.AddRenderer(self.renderer)
        self.renderInteractor.SetRenderWindow(self.renderWin)
    
        self.renderer.AddVolume(self.Volume())

        self.renderer.SetBackground(self.colors.GetColor3d("BkgColor"))
        self.renderWin.SetSize(550,550)
        self.renderWin.SetMultiSamples(4)

        self.renderWin.Render()
        self.renderInteractor.Start()

class VTKVisualiser():
    def __init__(self, data):
        self.data = data
        self.colors = vtk.vtkNamedColors()
        self.colors.SetColor('bgColor', [51, 77, 102, 255])

    def NiftiImport(self):
        self.niftiImporter = vtk.vtkNIFTIImageReader()
        # self.niftiImporter.setI

    def ConvertData(self):
        self.vtk_array = numpy_support.numpy_to_vtk(num_array=self.data.ravel(), deep=1, array_type=vtk.VTK_INT)
        print(self.vtk_array)

        self.noSlices = self.data.shape[2]
        self.data_stacked = np.dstack([self.data]*self.noSlices)

        x = np.arange(0, self.data.shape[0]+1)
        y = np.arange(0, self.data.shape[1]+1)
        z = np.arange(0, self.noSlices+1)

        gridToVTK('./data', x, y, z, cellData={'data': self.data_stacked})

        # self.volume = vtk.vtkImplicitVolume()

    def VTKDataImport(self):
        self.img_vtk = vtk.vtkImageData()
        self.img_vtk.SetDimensions(self.data.shape)
        self.img_vtk.SetSpacing(1,1,1)
        self.img_vtk.GetPointData().SetScalars(self.vtk_array)
        print(self.img_vtk)

        return self.img_vtk
    def VTKVolume(self):
        self.volume = vtk.vtkVolume()
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()

        self.volumeProperty.SetColor(self.Color()[1])
        self.volumeProperty.SetScalarOpacity(self.Color()[0])

        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)
        print(self.VTKVolume)
        return self.VTKVolume

    def DataImport(self):
        self.dataImporter = vtk.vtkImageImport()

        w,d,h = self.data.shape
        # self.dataImporterString = self.data.tostring()
        # self.dataImporter.CopyImportVoidPointer(self.dataImporterString, len(self.dataImporterString))
        self.dataImporter.CopyImportVoidPointer(self.data, self.data.nbytes)
        self.dataImporter.SetDataScalarTypeToUnsignedChar()
        self.dataImporter.SetNumberOfScalarComponents(1)
        self.dataImporter.SetDataSpacing(1.0, 1.0, 1.0)
        self.dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
        self.dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
        # print(self.dataImporter.GetOutput())
        return self.dataImporter

    def Color(self):
        self.opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self.colorTransferFunction = vtk.vtkColorTransferFunction()

        self.colorTransferFunction.AddRGBPoint(500, 1.0, 0.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(1000, 0.0, 1.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(1150, 0.0, 0.0, 1.0)

        self.opacityTransferFunction.AddPoint(0, 10.0)

        return self.opacityTransferFunction, self.colorTransferFunction

    def Volume(self):
        self.volume = vtk.vtkVolume()
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()

        self.volumeProperty.SetColor(self.Color()[1])
        self.volumeProperty.SetScalarOpacity(self.Color()[0])
        # self.volumeProperty.ShadeOn()

        self.volumeMapper.SetInputConnection(self.DataImport().GetOutputPort())
        # self.volumeMapper.SetMaximumImageSampleDistance(0.1)

        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)       
        print(self.volume) 
        return self.volume

    def Threshold(self):
        self.threshold = vtk.vtkImageThreshold()
        
        self.threshold.SetInputConnection(self.DataImport().GetOutputPort())

        self.threshold.ThresholdByLower(50)
        self.threshold.ReplaceInOn()
        self.threshold.SetInValue(0)  # set all values below 400 to 0
        self.threshold.ReplaceOutOn()
        self.threshold.SetOutValue(1)  # set all values above 400 to 1
        # self.threshold.Update()
        # print(self.threshold)
        return self.threshold
    
    def MarchingCubes(self):
        self.partExtractor = vtk.vtkMarchingCubes()
        self.partStripper = vtk.vtkStripper()
        self.partMapper = vtk.vtkPolyDataMapper()

        # self.partExtractor.SetInputConnection(self.DataImport().GetOutputPort())
        self.partExtractor.SetInputConnection(self.Threshold().GetOutputPort())
        self.partExtractor.ComputeNormalsOn()
        self.partExtractor.SetValue(0, 1)

        self.partStripper.SetInputConnection(self.partExtractor.GetOutputPort())

        self.partMapper.SetInputConnection(self.partStripper.GetOutputPort())
        self.partMapper.ScalarVisibilityOff()

        self.part = vtk.vtkActor()
        self.part.SetMapper(self.partMapper)
        self.part.GetProperty().SetDiffuseColor(self.colors.GetColor3d('Ivory'))

        return self.part

    def DiscreteMarchingCubes(self):
        self.partExtractor = vtk.vtkDiscreteMarchingCubes()
        self.partMapper = vtk.vtkPolyDataMapper()

        self.partExtractor.SetInputConnection(self.Threshold().GetOutputPort())
        self.partExtractor.GenerateValues(1,1,1)
        self.partExtractor.Update()

        self.partMapper.SetInputConnection(self.partExtractor.GetOutputPort())

        self.part = vtk.vtkActor()
        self.part.SetMapper(self.partMapper)
        self.part.GetProperty().SetDiffuseColor(self.colors.GetColor3d('Ivory'))

        self.partExtractor.Update()

        return self.part

    def Outline(self):
        self.outlineData = vtk.vtkOutlineFilter()
        self.outlineData.SetInputConnection(self.DataImport().GetOutputPort())

        self.mapOutline = vtk.vtkPolyDataMapper()
        self.mapOutline.SetInputConnection(self.outlineData.GetOutputPort())
        
        self.outline = vtk.vtkActor()
        self.outline.SetMapper(self.mapOutline)
        self.outline.GetProperty().SetColor(self.colors.GetColor3d('Red'))

        return self.outline

    def InitialiseMC(self, mctype = 'mc'):
        self.aRenderer = vtk.vtkRenderer()
        self.arenWin = vtk.vtkRenderWindow()
        self.aCamera = vtk.vtkCamera()
        self.iren = vtk.vtkRenderWindowInteractor()
        
        self.arenWin.AddRenderer(self.aRenderer)
        
        self.iren.SetRenderWindow(self.arenWin)

        self.aCamera.SetViewUp(0, 0, -1)
        self.aCamera.SetPosition(0, -1, 0)
        self.aCamera.SetFocalPoint(0, 0, 0)
        self.aCamera.ComputeViewPlaneNormal()
        self.aCamera.Azimuth(30.0)
        self.aCamera.Elevation(30.0)
        
        self.aRenderer.AddActor(self.Outline())
        if mctype == 'dmc':
            self.aRenderer.AddActor(self.DiscreteMarchingCubes())
        else:
            self.aRenderer.AddActor(self.MarchingCubes())
        self.aRenderer.SetActiveCamera(self.aCamera)
        self.aRenderer.ResetCamera()
        self.aCamera.Dolly(1.5)

        self.aRenderer.SetBackground(self.colors.GetColor3d("BkgColor"))
        self.arenWin.SetSize(640, 480)

        self.aRenderer.ResetCameraClippingRange()

        self.iren.Initialize()
        self.iren.Start()


    def InitialiseStack(self, voltype='nc'):
        self.renderer = vtk.vtkRenderer()
        self.renderWin = vtk.vtkRenderWindow()
        self.renderInteractor = vtk.vtkRenderWindowInteractor()

        self.renderWin.AddRenderer(self.renderer)
        self.renderInteractor.SetRenderWindow(self.renderWin)

        if voltype == 'convert':
            self.renderer.AddVolume(self.VTKVolume())
        else:
            self.renderer.AddVolume(self.Volume())

        self.renderer.SetBackground(self.colors.GetColor3d("BkgColor"))
        self.renderWin.SetSize(550,550)
        self.renderWin.SetMultiSamples(4)

        self.renderWin.Render()
        self.renderInteractor.Start()

    # threshold = vtk.vtkImageThreshold()
    # threshold.SetInputConnection(dataImporter.GetOutputPort())
    # threshold.ThresholdByLower(128)
    # threshold.ReplaceInOn()
    # threshold.SetInValue(0)
    # threshold.ReplaceOutOn()
    # threshold.SetOutValue(1)
    # threshold.Update()

    # dmc = vtk.vtkDiscreteMarchingCubes()
    # dmc.SetInputConnection(threshold.GetOutputPort())
    # dmc.GenerateValues(1,1,1)
    # dmc.Update()

class ModelProcessor():
    def __init__(self, directory):
        self.directory = directory
    
    # def get_program_parameters(self):
    #     import argparse
    #     self.description = 'Read a .stl file.'
    #     self.epilogue = ''''''
    #     self.parser = argparse.ArgumentParser(description=self.description, epilog=self.epilogue,
    #                                           formatter_class=argparse.RawDescriptionHelpFormatter)
    #     self.parser.add_argument('filename', help=self.filename)
    #     self.args = self.parser.parse_args()
    #     return self.args.filename

    def Colors(self):
        self.colors = vtk.vtkNamedColors()
        return self.colors

    def STLReader(self):
        self.reader = vtk.vtkSTLReader()
        self.reader.SetFileName(os.path.join(*self.directory.split(',')))
        return self.reader

    def Mapper(self):
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.STLReader().GetOutputPort())
        return self.mapper

    def Actor(self):
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.Mapper())
        return self.actor

    def Renderer(self):
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.ren.SetBackground(self.Colors().GetColor3d("black"))

        # Create a renderwindowinteractor
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        # Assign actor to the renderer
        self.ren.AddActor(self.Actor())

        # Enable user interface interactor
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

if __name__ == "__main__":
    directory = 'data,sample1,25,*.bmp'
    directory2 = 'data,sample1,25'
    directory3 = 'data,sample1,25.stl'
    stack = ImageProcessor(directory)
    stack.ImageImport()
    stack.CreateVolume()
    # plotHeatmap(stack.CreateVolume())
    # stack.ShowImage()
    # stack.CreateNifti()
    vtkvis = VTKVisualiser(stack.CreateVolume())
    # vtkvis.InitialiseStack(voltype='convert')
    # print(stack.CreateVolume().shape)
    # vtkvis.ShowThresholdImage()
    # vtkvis.ShowImportedImage()
    # vtkvis.ShowModifiedImage()
    # vtkvis.plotHeatMap(np.rot90(stack.CreateVolume()[:,:,56]))#[stack.CreateVolume().shape[0], :, :])
    # vtkvis.InitialiseStack()
    # vtkvis.plotHeatMap()
    # vtkvis.InitialiseMC()
    # vtkvis.InitialiseMC('dmc')
    # test = main(stack.CreateVolume())

    # vtkvis.ConvertData()
    newstack = ImageProcessor2(directory2)
    # newstack.BMPImageReader()
    newstack.InitialiseStack()
    
    # stlmodel = ModelProcessor(directory3)
    # stlmodel.Renderer()