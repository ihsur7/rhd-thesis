# import numpy as np
# import vtk
# from vtk.util import numpy_support
# import os
# import glob
# from PIL import Image

# import plotly
# from plotly.graph_objs import *

# data = np.arange(4*4*3).reshape(4,4,3)

# np.ravel(data)

# # print(data)

# directory = 'data,sample1,25,*.bmp'

# def ImageImport(directory):
#     img_list = []
#     # print(directory.split(','))
#     for imgfile in glob.glob(os.path.join(*directory.split(','))):
#         img = Image.open(imgfile)
#         img_list.append(img)
#     temp = img_list[0]
#     arr = np.array(temp)
#     w, d = temp.size
#     h = len(img_list)
#     # print('w,d,h: ', w,d,h)
#     volume = np.zeros((w,d,h),dtype=np.uint16)
#     imgslice = 0
#     for im in img_list:
#         imdata = list(im.getdata())
#         n = w
#         imdata1 = [imdata[i: i+n] for i in range(0, len(imdata), n)]
#         for i, x in enumerate(imdata1):
#             for j,y in enumerate(x):
#                 volume[i][j][imgslice] = y
#         imgslice += 1
#     # print(len(list(temp.getdata())))
#     # print(volume[:,:,-1])
#     return volume


    
# stack = ImageImport(directory)
# numpydatashape = stack.shape
# # print(stack[0])

# reader = vtk.vtkImageImport()
# w,d,h = stack.shape
# reader.CopyImportVoidPointer(stack[0], stack.nbytes)
# reader.SetDataScalarTypeToUnsignedChar()
# reader.SetNumberOfScalarComponents(1)
# reader.SetDataSpacing(1.0, 1.0, 1.0)
# reader.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
# reader.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)

# def vtkImageToNumPy(image, pixelDims):
#     pointData = image.GetPointData()
#     arrayData = pointData.GetArray(0)
#     ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
#     ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')
    
#     return ArrayDicom

# ConstPixelDims = numpydatashape

# ArrayDicom = vtkImageToNumPy(reader.GetOutput(), ConstPixelDims)

# def plotHeatmap(array, name="plot"):
#     data = Data([
#         Heatmap(
#             z=array,
#             scl='Greys'
#         )
#     ])
#     layout = Layout(
#         autosize=False,
#         title=name
#     )
#     fig = Figure(data=data, layout=layout)

#     return plotly.plotly.iplot(fig, filename=name)

# plotHeatmap(np.rot90(ArrayDicom[:, 56, :]), name="CT_Original")


directory2 = 'data,sample1,25'

directory3 = directory2 + ',*.bmp'

print(directory3)

