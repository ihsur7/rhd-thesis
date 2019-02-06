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

class VTKVisualiser():
    def __init__(self, data):
        self.data = data
        self.colors = vtk.vtkNamedColors()
        self.colors.SetColor('bgColor', [51, 77, 102, 255])

    def NiftiImport(self):
        self.niftiImporter = vtk.vtkNIFTIImageReader()
        # self.niftiImporter.setI
    
    def CreateVTKData(self):
        pass

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

    def plotHeatMap(self, array, name='plot'):
        self.trace = go.Heatmap(z=array)
        # self.hmdata = Data([
        #     Heatmap(z=array,
        #     scl = 'Greys'
        #     )
        # ])
        # self.layout = Layout(
        #     autosize=False,
        #     title=name
        # )
        self.hmdata = [self.trace]
        # self.fig = Figure(data=self.hmdata, layout = layout)

        return py.iplot(self.hmdata, filename=name)

    def Color(self):
        self.opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self.colorTransferFunction = vtk.vtkColorTransferFunction()

        # for i in range(0, 256):
        #     # self.opacityTransferFunction.AddPoint(i, 0.2)
        #     self.colorTransferFunction.AddRGBPoint(i, i/255.0, i/255.0, i/255.0)

        # self.opacityTransferFunction.AddPoint(0, 0)
        # self.colorTransferFunction.AddRGBPoint(0, 0, 0, 0)

        self.colorTransferFunction.AddRGBPoint(500, 1.0, 0.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(1000, 0.0, 1.0, 0.0)
        self.colorTransferFunction.AddRGBPoint(1150, 0.0, 0.0, 1.0)

        self.opacityTransferFunction.AddPoint(0, 10.0)

        # self.opacityTransferFunction.AddPoint(0, 0.0)
        # self.opacityTransferFunction.AddPoint(50, 0.05)
        # self.opacityTransferFunction.AddPoint(120, 0.1)
        # self.opacityTransferFunction.AddPoint(150, 1.0)

        # self.colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        # self.colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        # self.colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
        # self.colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
        # self.colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

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

    def vtkImageToNumpy(self, image, pixelDims):
        self.pointData = image.GetPointData()
        self.arrayData = self.pointData.GetArray(0)
        self.arrayImage = numpy_support.vtk_to_numpy(self.arrayData)
        self.arrayImage = self.arrayImage.reshape(pixelDims, order='F')
        return self.arrayImage
    
    def ShowImportedImage(self, slice = 0):
        self.arraydata = numpy_support.numpy_to_vtk(self.data.ravel(), deep=True, array_type=vtk.VTK_INT)
        self.img_vtk = vtk.vtkImageData()
        self.img_vtk.SetDimensions(self.data.shape)
        self.img_vtk.SetSpacing(1, 1, 1)
        self.img_vtk.GetPointData().SetScalars(self.arraydata)
        self.vtk_img = numpy_support.vtk_to_numpy(self.img_vtk)
        self.image = Image.fromarray(self.vtk_img)
        self.image.show()
        # self.arrayimage = self.vtkImageToNumpy(self.DataImport().GetOutput(), self.data.shape)
        # self.image = Image.fromarray(self.arrayimage)
        # print(self.arrayimage)
        return

    def ShowModifiedImage(self, slice = 0):
        # print(self.Threshold().GetOutputPort())
        print(self.data.shape)
        # self.nparray = self.vtkImageToNumpy(self.Threshold().GetOutput(), self.data.shape)
        self.vtkdata = self.Threshold().GetOutput()
        print(self.vtkdata)
        self.pointdata = self.vtkdata.GetPointData().GetScalars()
        print(self.pointdata)
        # self.arraydata = self.pointdata.GetArray(0)
        self.nparray = numpy_support.vtk_to_numpy(self.pointdata)
        print(self.nparray)

        # self.nparray = numpy_support.vtk_to_numpy(self.Threshold().GetOutput())
        self.img = Image.fromarray(self.nparray[:,:,slice], 'L')
        # array = self.CreateVolume().astype('uint8')
        # img = Image.fromarray(array[:, :, slice], 'L')
        self.img.show()

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


    def InitialiseStack(self):
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

if __name__ == "__main__":
    directory = 'data,sample1,25,*.bmp'
    stack = ImageProcessor(directory)
    stack.ImageImport()
    stack.CreateVolume()
    # plotHeatmap(stack.CreateVolume())
    # stack.ShowImage()
    # stack.CreateNifti()
    vtkvis = VTKVisualiser(stack.CreateVolume())
    print(stack.CreateVolume().shape)
    vtkvis.ShowImportedImage()
    # vtkvis.ShowModifiedImage()
    # vtkvis.plotHeatMap(np.rot90(stack.CreateVolume()[:,:,56]))#[stack.CreateVolume().shape[0], :, :])
    # vtkvis.InitialiseStack()
    # vtkvis.plotHeatMap()
    # vtkvis.InitialiseMC()
    # vtkvis.InitialiseMC('dmc')
    # test = main(stack.CreateVolume())
    