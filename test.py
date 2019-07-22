# a = [False, True, False]
# a= list(filter(lambda x: x == True, a))
# print(a)

# a = '0-lx.tif'
# print(a[:-4])

# class Data():
#     def __init__(self):
#         self.data = {}

# a = Data()
# print(type(a.data), a.data)

# a.data["raw_data"] = range(5)
# print(a.data)

import scipy as sp
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

size = (100,100)

im = np.random.randint(0, 2, size=size)
dt = ndimage.distance_transform_edt(im > 0)
dt2 = ndimage.distance_transform_edt(im)
print(im)
print(dt)
print(dt2)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[5,5], constrained_layout=True)
for i,j in zip(ax.flat, [im, dt, dt2]):
    im2 = i.imshow(j)
fig.colorbar(im2, ax=ax.flat)

# plt.tight_layout(True)
plt.show()

'''
def main(stack):
    colors = vtk.vtkNamedColors()

    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)


    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(stack, stack.nbytes)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)

    w,d,h = stack.shape
    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)

    # dataImporter.SetDataSpacing(0.625, 0.48, 0.48)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(20, 0.0)
    opacityTransferFunction.AddPoint(255, 0.2)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(1,1,1)
    # ren1.SetBackground(colors.GetColor3d("Wheat"))
    # ren1.GetActiveCamera().Azimuth(45)
    # ren1.GetActiveCamera().Elevation(30)
    renWin.Render()
    iren.Start()

    '''