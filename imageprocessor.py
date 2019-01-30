import os
import sys
import matplotlib.pyplot as plt
import glob
import vtk
from vtk.util import numpy_support
from vtk.util.misc import vtkGetDataRoot
import numpy as np
from PIL import Image
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)

print(vtk.vtkVersion.GetVTKSourceVersion())

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

class VTKVisualiser():
    def __init__(self, data):
        self.data = data
        self.colors = vtk.vtkNamedColors()
        self.colors.SetColor('bgColor', [51, 77, 102, 255])

    def DataImport(self):
        self.dataImporter = vtk.vtkImageImport()

        w,d,h = self.data.shape
        self.dataImporter.CopyImportVoidPointer(self.data, self.data.nbytes)
        self.dataImporter.SetDataScalarTypeToUnsignedChar()
        self.dataImporter.SetNumberOfScalarComponents(1)
        self.dataImporter.SetDataSpacing(1,1,1)
        self.dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
        self.dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
        
        self.dataImporter.Update()

        return self.dataImporter

    def Color(self):
        self.opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self.colorTransferFunction = vtk.vtkColorTransferFunction()

        # for i in range(0, 256):
        #     # self.opacityTransferFunction.AddPoint(i, 0.2)
        #     self.colorTransferFunction.AddRGBPoint(i, i/255.0, i/255.0, i/255.0)

        # self.opacityTransferFunction.AddPoint(0, 0)
        # self.colorTransferFunction.AddRGBPoint(0, 0, 0, 0)

        self.colorTransferFunction.AddRGBPoint(500, 1.0, 0.5, 0.3)
        self.colorTransferFunction.AddRGBPoint(1000, 1.0, 0.5, 0.3)
        self.colorTransferFunction.AddRGBPoint(1150, 1.0, 1.0, 0.9)

        self.opacityTransferFunction.AddPoint(10, 0.0)
        self.opacityTransferFunction.AddPoint(100, 0.5)
        self.opacityTransferFunction.AddPoint(500, 1.0)

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
        self.volumeProperty.ShadeOn()

        self.volumeMapper.SetInputConnection(self.DataImport().GetOutputPort())
        # self.volumeMapper.SetMaximumImageSampleDistance(0.1)

        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)

        return self.volume

    def Threshold(self):
        self.threshold = vtk.vtkImageThreshold()
        
        self.threshold.SetInputConnection(self.DataImport().GetOutputPort())

        self.threshold.ThresholdByLower(254)  # remove all soft tissue
        self.threshold.ReplaceInOn()
        self.threshold.SetInValue(0)  # set all values below 400 to 0
        self.threshold.ReplaceOutOn()
        self.threshold.SetOutValue(1)  # set all values above 400 to 1
        self.threshold.Update()
        
        return self.threshold

    def MarchingCubes(self):
        self.partExtractor = vtk.vtkMarchingCubes()
        self.partStripper = vtk.vtkStripper()
        self.partMapper = vtk.vtkPolyDataMapper()

        self.partExtractor.SetInputConnection(self.DataImport().GetOutputPort())
        self.partExtractor.ComputeNormalsOn()
        self.partExtractor.SetValue(0, 254)

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
    vtkvis = VTKVisualiser(stack.CreateVolume())
    vtkvis.InitialiseStack()
    # vtkvis.InitialiseMC()
    # vtkvis.InitialiseMC('dmc')
    # test = main(stack.CreateVolume())
    